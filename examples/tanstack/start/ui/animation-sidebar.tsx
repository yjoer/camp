import * as stylex from '@stylexjs/stylex';
import { createFileRoute } from '@tanstack/react-router';
import { useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import { button_styles } from '@/components/button';

export const Route = createFileRoute('/ui/animations/sidebar')({
  component: AnimationSidebar,
});

function AnimationSidebar() {
  return (
    <div className="mx-2 my-1 flex flex-col gap-4">
      <SidebarTransition />
      <SidebarWAAPI />
    </div>
  );
}

function SidebarTransition() {
  const ref = useRef<HTMLDivElement | null>(null);
  const [state, set_state] = useState<'closed' | 'closing' | 'opened' | 'opening'>('closed');

  const handle_toggle = () => {
    if (state === 'closed' || state == 'closing') set_state('opening');
    else set_state('closing');
  };

  useLayoutEffect(() => {
    if (!ref.current) return;
    const el = ref.current;

    if (state !== 'opening' && state !== 'closing') return;

    if (state === 'opening') {
      el.style.transition = 'transform 250ms';
      void el.clientWidth; // reflow
      el.style.transform = 'translateX(0)';
    } else if (state === 'closing') {
      el.style.transform = 'translateX(100%)';
    }

    const handle_transition_end = () => {
      if (state === 'opening') set_state('opened');
      if (state === 'closing') set_state('closed');
    };

    el.addEventListener('transitionend', handle_transition_end);
    return () => el.removeEventListener('transitionend', handle_transition_end);
  }, [state]);

  return (
    <div>
      <span className="bg-[#ffa500]">CSS Transform</span>
      <div className="mt-1">
        <button onClick={handle_toggle} {...stylex.props(button_styles.base)}>
          Toggle
        </button>
      </div>
      {state !== 'closed' &&
        createPortal(
          <div className="pointer-events-none absolute inset-0 overflow-hidden">
            <Sidebar ref={ref} />
          </div>,
          document.body,
        )}
    </div>
  );
}

function SidebarWAAPI() {
  const ref = useRef<HTMLDivElement | null>(null);
  const animation_ref = useRef<Animation | null>(null);
  const [state, set_state] = useState<'closed' | 'closing' | 'opened' | 'opening'>('closed');

  const handle_toggle = () => {
    if (state === 'closed' || state === 'closing') set_state('opening');
    else set_state('closing');
  };

  useLayoutEffect(() => {
    if (!ref.current) return;
    const el = ref.current;

    if (state !== 'opening' && state !== 'closing') return;

    const effect = animation_ref.current?.effect as KeyframeEffect | undefined;
    if (!animation_ref.current || effect?.target !== el) {
      const animation = el.animate(
        [{ transform: 'translateX(100%)' }, { transform: 'translateX(0)' }],
        {
          duration: 250,
          easing: 'ease-in-out',
          fill: 'forwards',
        },
      );

      animation.pause();
      animation_ref.current = animation;
    }

    animation_ref.current.playbackRate = state === 'opening' ? 1 : -1;
    animation_ref.current.play();

    animation_ref.current.finished
    .then(() => {
      if (state === 'opening') set_state('opened');
      if (state === 'closing') set_state('closed');
    })
    .catch(() => {});
  }, [state]);

  return (
    <div>
      <span className="bg-[#ffa500]">WAAPI Transform</span>
      <div className="mt-1">
        <button onClick={handle_toggle} {...stylex.props(button_styles.base)}>
          Toggle
        </button>
      </div>
      {state !== 'closed' &&
        createPortal(
          <div className="pointer-events-none absolute inset-0 overflow-hidden">
            <Sidebar ref={ref} />
          </div>,
          document.body,
        )}
    </div>
  );
}

interface SidebarProps {
  ref: React.Ref<HTMLDivElement>;
}

function Sidebar({ ref }: SidebarProps) {
  return (
    <div ref={ref} {...stylex.props(sidebar_sx.base)}>
      {Array.from({ length: 5 }).map((_, idx) => (
        <div key={idx} className="flex gap-2 px-2 py-1">
          <div {...stylex.props(sidebar_sx.icon)} />
          <div {...stylex.props(sidebar_sx.label)} />
        </div>
      ))}
    </div>
  );
}

const sidebar_sx = stylex.create({
  base: {
    position: 'absolute',
    top: 0,
    right: 0,
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    borderLeftColor: 'oklch(92% 0 0 / 1)',
    borderLeftStyle: 'solid',
    borderLeftWidth: 1,
    transform: 'translateX(100%)',
  },
  icon: {
    width: 24,
    height: 24,
    backgroundColor: 'oklch(92% 0 0)',
  },
  label: {
    width: 240,
    height: 24,
    backgroundColor: 'oklch(92% 0 0)',
  },
});
