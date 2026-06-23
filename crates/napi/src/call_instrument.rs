//! A generic hook for instrumenting `#[napi(instrument)]` calls, with no coupling to any particular
//! tracing or telemetry stack.
//!
//! napi itself stays agnostic about *how* a call is instrumented: it only provides the seam. A host
//! registers a [`CallInstrument`] once at startup; thereafter, every `#[napi(instrument)]` function
//! asks it for an optional [`CallGuard`] on the JS thread at call entry (where the host can read the
//! caller's ambient context, e.g. an OpenTelemetry context), and the guard is entered for the
//! duration of the call — across every poll, for async functions.
//!
//! When no hook is registered the cost is a single atomic load plus a `None` branch, so
//! leaving the `instrument` feature compiled in is cheap. Functions *without* `#[napi(instrument)]`
//! emit no calls into this module at all.

use std::any::Any;
use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;
use std::task::{Context, Poll};

use pin_project_lite::pin_project;

use crate::sys;

/// A per-call guard. [`enter`](CallGuard::enter) is invoked each time the call is "active" — once for
/// a synchronous call, and on every poll for an async call — returning an inner guard that is dropped
/// when the active region ends (e.g. a `tracing` span's entered guard).
pub trait CallGuard: Send {
  fn enter(&self) -> Box<dyn Any>;
}

/// Host-provided instrumentation. Invoked on the JS thread at the entry of every `#[napi(instrument)]`
/// call, before any work is offloaded to a runtime thread, so the implementation can read JS-side
/// ambient state via `env`. Returning `None` skips instrumentation for that call.
pub trait CallInstrument: Send + Sync {
  fn on_call(&self, env: sys::napi_env, name: &'static str) -> Option<Box<dyn CallGuard>>;
}

static HOOK: OnceLock<&'static dyn CallInstrument> = OnceLock::new();

/// Register the process-wide instrumentation hook. The first registration wins; later calls are
/// no-ops, so this is safe to call defensively from startup.
pub fn register_call_instrument(hook: &'static dyn CallInstrument) {
  let _ = HOOK.set(hook);
}

#[inline]
fn current_hook() -> Option<&'static dyn CallInstrument> {
  HOOK.get().copied()
}

/// Open a guard for a synchronous `#[napi(instrument)]` call. The returned value must be held for the
/// body of the call; dropping it ends the active region.
#[inline]
pub fn enter_sync(env: sys::napi_env, name: &'static str) -> Option<Box<dyn Any>> {
  current_hook()
    .and_then(|hook| hook.on_call(env, name))
    .map(|guard| guard.enter())
}

/// Wrap an async `#[napi(instrument)]` future so the guard is entered around every poll. The guard is
/// acquired eagerly (on the JS thread) so the host can capture ambient context before the future is
/// handed to a runtime.
#[inline]
pub fn instrument<F: Future>(
  env: sys::napi_env,
  name: &'static str,
  future: F,
) -> InstrumentedCall<F> {
  InstrumentedCall {
    future,
    guard: current_hook().and_then(|hook| hook.on_call(env, name)),
  }
}

pin_project! {
  /// Future returned by [`instrument`]. Enters the guard (if any) around each poll of the inner future.
  pub struct InstrumentedCall<F> {
    #[pin]
    future: F,
    guard: Option<Box<dyn CallGuard>>,
  }
}

impl<F: Future> Future for InstrumentedCall<F> {
  type Output = F::Output;

  fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
    let this = self.project();
    let _entered = this.guard.as_ref().map(|guard| guard.enter());
    this.future.poll(cx)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::sync::atomic::{AtomicUsize, Ordering};
  use std::task::{RawWaker, RawWakerVTable, Waker};

  static ENTERS: AtomicUsize = AtomicUsize::new(0);

  struct CountingGuard;
  impl CallGuard for CountingGuard {
    fn enter(&self) -> Box<dyn Any> {
      ENTERS.fetch_add(1, Ordering::Relaxed);
      Box::new(())
    }
  }

  struct CountingInstrument;
  impl CallInstrument for CountingInstrument {
    fn on_call(&self, _env: sys::napi_env, _name: &'static str) -> Option<Box<dyn CallGuard>> {
      Some(Box::new(CountingGuard))
    }
  }

  fn noop_waker() -> Waker {
    fn no_op(_: *const ()) {}
    fn clone(_: *const ()) -> RawWaker {
      RawWaker::new(std::ptr::null(), &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);
    unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
  }

  // `HOOK` is a write-once process global, so the no-hook case is only observable before any
  // registration; both cases share one test rather than relying on inter-test ordering.
  #[test]
  fn hook_lifecycle() {
    // Before any registration, guard acquisition is a no-op.
    assert!(enter_sync(std::ptr::null_mut(), "x").is_none());

    static INSTRUMENT: CountingInstrument = CountingInstrument;
    register_call_instrument(&INSTRUMENT);

    let mut polls = 0u32;
    let fut = std::future::poll_fn(move |_cx| {
      polls += 1;
      if polls < 3 {
        Poll::Pending
      } else {
        Poll::Ready(polls)
      }
    });
    let mut wrapped = Box::pin(instrument(std::ptr::null_mut(), "f", fut));

    let waker = noop_waker();
    let mut cx = Context::from_waker(&waker);
    let before = ENTERS.load(Ordering::Relaxed);
    assert!(matches!(wrapped.as_mut().poll(&mut cx), Poll::Pending));
    assert!(matches!(wrapped.as_mut().poll(&mut cx), Poll::Pending));
    assert!(matches!(wrapped.as_mut().poll(&mut cx), Poll::Ready(3)));
    // Entered once per poll.
    assert_eq!(ENTERS.load(Ordering::Relaxed) - before, 3);
  }
}
