/* eslint-disable react-hooks/exhaustive-deps */
import { useCallback, useEffect, useRef, useState } from 'react';

export const Route = createFileRoute({
  component: StaleClosures,
});

function StaleClosures() {
  return (
    <div className="mx-2 my-1 flex flex-col gap-4">
      <DependencyArray />
      <RefSync />
      <StateRefHook />
    </div>
  );
}

function DependencyArray() {
  const timedLogRef = useRef<HTMLDivElement>(null!);
  const timedLogStaleRef = useRef<HTMLDivElement>(null!);
  const logRef = useRef<HTMLDivElement>(null!);
  const logStaleRef = useRef<HTMLDivElement>(null!);

  const [count, setCount] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      timedLogRef.current.textContent = `Timed Log: ${count}`;
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, [count]);

  useEffect(() => {
    const interval = setInterval(() => {
      timedLogStaleRef.current.textContent = `Timed Log (Stale): ${count}`;
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  const handleClick = useCallback(() => {
    logRef.current.textContent = `Log: ${count}`;
  }, [count]);

  const handleClickStale = useCallback(() => {
    logStaleRef.current.textContent = `Log (Stale): ${count}`;
  }, []);

  return (
    <div>
      <span className="bg-[#ffdd00]">Dependency Array</span>
      <div>Count: {count}</div>
      <div ref={timedLogRef}>Timed Log:</div>
      <div ref={timedLogStaleRef}>Timed Log (Stale):</div>
      <div ref={logRef}>Log:</div>
      <div ref={logStaleRef}>Log (Stale):</div>
      <div className="mt-1 flex gap-2">
        <button className="font-semibold" onClick={() => setCount((prev) => prev + 1)}>
          Increment
        </button>
        <button className="font-semibold" onClick={handleClick}>
          Log
        </button>
        <button className="font-semibold" onClick={handleClickStale}>
          Log (Stale)
        </button>
      </div>
    </div>
  );
}

function RefSync() {
  const [count, setCount] = useState(0);
  const countRef = useRef(count);

  const timedLogRef = useRef<HTMLDivElement>(null!);
  const logRef = useRef<HTMLDivElement>(null!);

  useEffect(() => {
    const interval = setInterval(() => {
      timedLogRef.current.textContent = `Timed Log: ${countRef.current}`;
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  const handleClick = useCallback(() => {
    logRef.current.textContent = `Log: ${countRef.current}`;
  }, []);

  const handleIncrement = useCallback(() => {
    setCount((prev) => prev + 1);
    countRef.current += 1;
  }, []);

  return (
    <div>
      <span className="bg-[#ffdd00]">Ref Sync</span>
      <div>Count: {count}</div>
      <div ref={timedLogRef}>Timed Log:</div>
      <div ref={logRef}>Log:</div>
      <div className="mt-1 flex gap-2">
        <button className="font-semibold" onClick={handleIncrement}>
          Increment
        </button>
        <button className="font-semibold" onClick={handleClick}>
          Log
        </button>
      </div>
    </div>
  );
}

function StateRefHook() {
  const [count, setCount] = useStateRef(0);

  const timedLogRef = useRef<HTMLDivElement>(null!);
  const logRef = useRef<HTMLDivElement>(null!);

  useEffect(() => {
    const interval = setInterval(() => {
      timedLogRef.current.textContent = `Timed Log: ${count.current}`;
    }, 1000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  const handleClick = useCallback(() => {
    logRef.current.textContent = `Log: ${count.current}`;
  }, []);

  const handleIncrement = useCallback(() => {
    setCount(count.current + 1);
  }, []);

  return (
    <div>
      <span className="bg-[#ffdd00]">State Ref Hook</span>
      <div>Count: {count.current}</div>
      <div ref={timedLogRef}>Timed Log:</div>
      <div ref={logRef}>Log:</div>
      <div className="mt-1 flex gap-2">
        <button className="font-semibold" onClick={handleIncrement}>
          Increment
        </button>
        <button className="font-semibold" onClick={handleClick}>
          Log
        </button>
      </div>
    </div>
  );
}

function useStateRef<T>(value: T): [React.RefObject<T>, (newState: T) => void] {
  const ref = useRef(value);
  const [, forceRender] = useState(false);

  function setState(newState: T) {
    if (Object.is(ref.current, newState)) return;

    ref.current = newState;
    forceRender((prev) => !prev);
  }

  return [ref, setState];
}
