use core::cell::UnsafeCell;
use core::fmt;
use core::marker::PhantomData;
use core::mem::{self, ManuallyDrop};
use core::ops::Deref;
use core::ptr;
use core::sync::atomic::{self, AtomicUsize, Ordering};
use crossbeam_utils::Backoff;
use guard::Guard;

/// Freeze as defined by `std`.
pub unsafe auto trait Freeze {}

impl<T: ?Sized> !Freeze for UnsafeCell<T> {}
unsafe impl<T: ?Sized> Freeze for PhantomData<T> {}
unsafe impl<T: ?Sized> Freeze for *const T {}
unsafe impl<T: ?Sized> Freeze for *mut T {}
unsafe impl<T: ?Sized> Freeze for &T {}
unsafe impl<T: ?Sized> Freeze for &mut T {}

/// An `AtomicCell` like type that works with types that support `Freeze`.
#[repr(transparent)]
pub struct Inline<T: ?Sized> {
    value: UnsafeCell<T>,
}

impl<T> fmt::Debug for Inline<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("Inline { .. }")
    }
}

unsafe impl<T: ?Sized + Send> Send for Inline<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for Inline<T> {}

impl<T> Inline<T> {
    /// Creates a new inline atomic initialized with `val`.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_epoch::Inline;
    ///
    /// let a = Inline::new(vec![1, 2, 3, 4]);
    /// ```
    pub const fn new(value: T) -> Self {
        Inline {
            value: UnsafeCell::new(value),
        }
    }

    /// Returns `true` if operations on values of this type are lock-free.
    pub fn is_lock_free() -> bool {
        atomic_is_lock_free::<T>()
    }

    /// Unwraps the inline atomic and returns its inner value.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_epoch::Inline;
    ///
    /// let mut a = Inline::new(vec![42]);
    /// let v = a.into_inner();
    ///
    /// assert_eq!(v, &[42]);
    /// ```
    pub fn into_inner(self) -> T {
        self.value.into_inner()
    }
}

impl<T: Freeze> Inline<T> {
    /// Loads a snapshot of the contained value.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_epoch::{self as epoch, Inline};
    ///
    /// let a = Inline::new(vec![7, 8, 9]);
    /// let guard = &epoch::pin();
    /// let p = a.load(guard);
    ///
    /// assert_eq!(*p, &[7, 8, 9]);
    /// ```
    pub fn load<'g>(&'g self, _: &'g Guard) -> Snapshot<'g, T> {
        let value = unsafe { atomic_load(self.value.get()) };
        Snapshot {
            value,
            _lifetime: PhantomData,
        }
    }
}

impl<T: 'static + Freeze + Send> Inline<T> {
    /// Stores a new value inline.
    ///
    /// # Examples
    ///
    /// ```
    /// use crossbeam_epoch::{self as epoch, Inline};
    ///
    /// let a = Inline::new(String::from("hello"));
    /// let guard = &epoch::pin();
    /// a.store("world".to_owned(), guard);
    ///
    /// assert_eq!(*a.load(guard), "world");
    /// ```
    pub fn store(&self, val: T, guard: &Guard) {
        if mem::needs_drop::<T>() {
            let old = unsafe { atomic_swap(self.value.get(), val) };
            guard.defer(move || {
                ManuallyDrop::into_inner(old);
            })
        } else {
            unsafe { atomic_store(self.value.get(), val) };
        }
    }
}

/// A non-owning snapshot of a value.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Snapshot<'a, T> {
    value: ManuallyDrop<T>,
    _lifetime: PhantomData<&'a T>,
}

impl<T> Deref for Snapshot<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Below this point, things are copied from AtomicCell
////////////////////////////////////////////////////////////////////////////////

/// An atomic `()`.
///
/// All operations are noops.
struct AtomicUnit;

impl AtomicUnit {
    #[inline]
    fn load(&self, _order: Ordering) {}

    #[inline]
    fn store(&self, _val: (), _order: Ordering) {}

    #[inline]
    fn swap(&self, _val: (), _order: Ordering) {}
}

macro_rules! atomic {
    // If values of type `$t` can be transmuted into values of the primitive atomic type `$atomic`,
    // declares variable `$a` of type `$atomic` and executes `$atomic_op`, breaking out of the loop.
    (@check, $t:ty, $atomic:ty, $a:ident, $atomic_op:expr) => {
        if can_transmute::<$t, $atomic>() {
            let $a: &$atomic;
            break $atomic_op;
        }
    };

    // If values of type `$t` can be transmuted into values of a primitive atomic type, declares
    // variable `$a` of that type and executes `$atomic_op`. Otherwise, just executes
    // `$fallback_op`.
    ($t:ty, $a:ident, $atomic_op:expr, $fallback_op:expr) => {
        loop {
            atomic!(@check, $t, AtomicUnit, $a, $atomic_op);
            atomic!(@check, $t, atomic::AtomicUsize, $a, $atomic_op);

            #[cfg(feature = "nightly")]
            {
                #[cfg(target_has_atomic = "8")]
                atomic!(@check, $t, atomic::AtomicU8, $a, $atomic_op);
                #[cfg(target_has_atomic = "16")]
                atomic!(@check, $t, atomic::AtomicU16, $a, $atomic_op);
                #[cfg(target_has_atomic = "32")]
                atomic!(@check, $t, atomic::AtomicU32, $a, $atomic_op);
                #[cfg(target_has_atomic = "64")]
                atomic!(@check, $t, atomic::AtomicU64, $a, $atomic_op);
            }

            break $fallback_op;
        }
    };
}

/// Returns `true` if operations on `AtomicCell<T>` are lock-free.
fn atomic_is_lock_free<T>() -> bool {
    atomic! { T, _a, true, false }
}

/// Returns `true` if values of type `A` can be transmuted into values of type `B`.
fn can_transmute<A, B>() -> bool {
    // Sizes must be equal, but alignment of `A` must be greater or equal than that of `B`.
    mem::size_of::<A>() == mem::size_of::<B>() && mem::align_of::<A>() >= mem::align_of::<B>()
}

/// A simple stamped lock.
struct Lock {
    /// The current state of the lock.
    ///
    /// All bits except the least significant one hold the current stamp. When locked, the state
    /// equals 1 and doesn't contain a valid stamp.
    state: AtomicUsize,
}

impl Lock {
    /// If not locked, returns the current stamp.
    ///
    /// This method should be called before optimistic reads.
    #[inline]
    fn optimistic_read(&self) -> Option<usize> {
        let state = self.state.load(Ordering::Acquire);
        if state == 1 {
            None
        } else {
            Some(state)
        }
    }

    /// Returns `true` if the current stamp is equal to `stamp`.
    ///
    /// This method should be called after optimistic reads to check whether they are valid. The
    /// argument `stamp` should correspond to the one returned by method `optimistic_read`.
    #[inline]
    fn validate_read(&self, stamp: usize) -> bool {
        atomic::fence(Ordering::Acquire);
        self.state.load(Ordering::Relaxed) == stamp
    }

    /// Grabs the lock for writing.
    #[inline]
    fn write(&'static self) -> WriteGuard {
        let backoff = Backoff::new();
        loop {
            let previous = self.state.swap(1, Ordering::Acquire);

            if previous != 1 {
                atomic::fence(Ordering::Release);

                return WriteGuard {
                    lock: self,
                    state: previous,
                };
            }

            backoff.snooze();
        }
    }
}

/// A RAII guard that releases the lock and increments the stamp when dropped.
struct WriteGuard {
    /// The parent lock.
    lock: &'static Lock,

    /// The stamp before locking.
    state: usize,
}

impl WriteGuard {
    /// Releases the lock without incrementing the stamp.
    #[inline]
    fn abort(self) {
        self.lock.state.store(self.state, Ordering::Release);
    }
}

impl Drop for WriteGuard {
    #[inline]
    fn drop(&mut self) {
        // Release the lock and increment the stamp.
        self.lock
            .state
            .store(self.state.wrapping_add(2), Ordering::Release);
    }
}

/// Returns a reference to the global lock associated with the `AtomicCell` at address `addr`.
///
/// This function is used to protect atomic data which doesn't fit into any of the primitive atomic
/// types in `std::sync::atomic`. Operations on such atomics must therefore use a global lock.
///
/// However, there is not only one global lock but an array of many locks, and one of them is
/// picked based on the given address. Having many locks reduces contention and improves
/// scalability.
#[inline]
#[must_use]
fn lock(addr: usize) -> &'static Lock {
    // The number of locks is a prime number because we want to make sure `addr % LEN` gets
    // dispersed across all locks.
    //
    // Note that addresses are always aligned to some power of 2, depending on type `T` in
    // `AtomicCell<T>`. If `LEN` was an even number, then `addr % LEN` would be an even number,
    // too, which means only half of the locks would get utilized!
    //
    // It is also possible for addresses to accidentally get aligned to a number that is not a
    // power of 2. Consider this example:
    //
    // ```
    // #[repr(C)]
    // struct Foo {
    //     a: AtomicCell<u8>,
    //     b: u8,
    //     c: u8,
    // }
    // ```
    //
    // Now, if we have a slice of type `&[Foo]`, it is possible that field `a` in all items gets
    // stored at addresses that are multiples of 3. It'd be too bad if `LEN` was divisible by 3.
    // In order to protect from such cases, we simply choose a large prime number for `LEN`.
    const LEN: usize = 97;

    const L: Lock = Lock {
        state: AtomicUsize::new(0),
    };
    static LOCKS: [Lock; LEN] = [
        L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L,
        L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L,
        L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L, L,
        L, L, L, L, L, L, L,
    ];

    // If the modulus is a constant number, the compiler will use crazy math to transform this into
    // a sequence of cheap arithmetic operations rather than using the slow modulo instruction.
    &LOCKS[addr % LEN]
}

/// Signature changed to not require copy
unsafe fn atomic_load<T: Freeze>(src: *mut T) -> ManuallyDrop<T> {
    atomic! {
        T, a,
        {
            a = &*(src as *const _ as *const _);
            mem::transmute_copy(&a.load(Ordering::Acquire))
        },
        {
            let lock = lock(src as usize);

            // Try doing an optimistic read first.
            if let Some(stamp) = lock.optimistic_read() {
                // We need a volatile read here because other threads might concurrently modify the
                // value. In theory, data races are *always* UB, even if we use volatile reads and
                // discard the data when a data race is detected. The proper solution would be to
                // do atomic reads and atomic writes, but we can't atomically read and write all
                // kinds of data since `AtomicU8` is not available on stable Rust yet.
                let val = ptr::read_volatile(src as *const _);

                if lock.validate_read(stamp) {
                    return val;
                }
            }

            // Grab a regular write lock so that writers don't starve this load.
            let guard = lock.write();
            let val = ptr::read(src as *const _);
            // The value hasn't been changed. Drop the guard without incrementing the stamp.
            guard.abort();
            val
        }
    }
}

/// Atomically writes `val` to `dst`.
///
/// This operation uses the `Release` ordering. If possible, an atomic instructions is used, and a
/// global lock otherwise.
unsafe fn atomic_store<T>(dst: *mut T, val: T) {
    atomic! {
        T, a,
        {
            a = &*(dst as *const _ as *const _);
            let res = a.store(mem::transmute_copy(&val), Ordering::Release);
            mem::forget(val);
            res
        },
        {
            let _guard = lock(dst as usize).write();
            ptr::write(dst, val)
        }
    }
}

/// Atomically swaps data at `dst` with `val`.
///
/// This operation uses the `AcqRel` ordering. If possible, an atomic instructions is used, and a
/// global lock otherwise.
unsafe fn atomic_swap<T>(dst: *mut T, val: T) -> ManuallyDrop<T> {
    atomic! {
        T, a,
        {
            a = &*(dst as *const _ as *const _);
            let res = mem::transmute_copy(&a.swap(mem::transmute_copy(&val), Ordering::AcqRel));
            mem::forget(val);
            res
        },
        {
            let _guard = lock(dst as usize).write();
            ptr::replace(dst as *mut _, ManuallyDrop::new(val))
        }
    }
}
