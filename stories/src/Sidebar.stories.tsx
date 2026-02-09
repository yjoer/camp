import { cva } from 'cva';
import { useRef, useState } from 'react';

import type { Meta, StoryObj } from '@storybook/react-vite';

const meta = {
  title: 'Components/Sidebar',
  component: SidebarMaxWidth,
} satisfies Meta<typeof SidebarMaxWidth>;

export default meta;

type Story = StoryObj<typeof meta>;

export const MaxWidth: Story = {
  args: {},
};

export const Width: Story = {
  render: () => {
    return <SidebarWidth />;
  },
};

export const WebAnimationAPI: Story = {
  render: () => {
    return <SidebarAnimationAPI />;
  },
};

function SidebarMaxWidth() {
  const sidebarRef = useRef<HTMLDivElement>(null!);
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = () => {
    sidebarRef.current.style.maxWidth = isOpen ? '0' : '300px';
    setIsOpen((prev) => !prev);
  };

  return (
    <div>
      <button className="font-semibold" onClick={handleToggle}>
        Toggle Sidebar
      </button>
      <Sidebar ref={sidebarRef} style={{ maxWidth: 0 }} type="max_width" />
    </div>
  );
}

function SidebarWidth() {
  const sidebarRef = useRef<HTMLDivElement>(null!);
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = () => {
    if (isOpen) {
      sidebarRef.current.style.width = '0';
    } else {
      sidebarRef.current.classList.remove('hidden');
      sidebarRef.current.classList.add('flex');
      const width = sidebarRef.current.clientWidth; // reflow
      sidebarRef.current.style.width = '0';

      void sidebarRef.current.offsetWidth; // reflow

      sidebarRef.current.style.width = `${width}px`;
    }

    setIsOpen((prev) => !prev);
  };

  const handleToggleRAF = () => {
    if (isOpen) {
      sidebarRef.current.style.width = '0';
    } else {
      sidebarRef.current.classList.remove('hidden');
      sidebarRef.current.classList.add('flex');
      const width = sidebarRef.current.clientWidth; // reflow
      sidebarRef.current.style.width = '0';

      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          sidebarRef.current.style.width = `${width}px`;
        });
      });
    }

    setIsOpen((prev) => !prev);
  };

  const handleTransitionEnd = () => {
    if (isOpen) return;
    sidebarRef.current.classList.add('hidden');
    sidebarRef.current.classList.remove('flex');
    sidebarRef.current.style.width = '';
  };

  return (
    <div>
      <div className="flex flex-col items-start gap-2">
        <button className="font-semibold" onClick={handleToggle}>
          Toggle Sidebar
        </button>
        <button className="font-semibold" onClick={handleToggleRAF}>
          Toggle Sidebar (rAF)
        </button>
      </div>
      <Sidebar ref={sidebarRef} type="width" onTransitionEnd={handleTransitionEnd} />
    </div>
  );
}

function SidebarAnimationAPI() {
  const sidebarRef = useRef<HTMLDivElement>(null!);
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = () => {
    if (!isOpen) {
      sidebarRef.current.classList.remove('hidden');
      sidebarRef.current.classList.add('flex');
    }

    const width = sidebarRef.current.clientWidth; // reflow
    const keyframes = [{ width: '0px' }, { width: `${width}px` }];

    if (isOpen) {
      const animation = sidebarRef.current.animate(keyframes.toReversed(), { duration: 250 });

      animation.onfinish = () => {
        sidebarRef.current.classList.add('hidden');
        sidebarRef.current.classList.remove('flex');
      };
    } else {
      sidebarRef.current.animate(keyframes, { duration: 250 });
    }

    setIsOpen((prev) => !prev);
  };

  return (
    <div>
      <button className="font-semibold" onClick={handleToggle}>
        Toggle Sidebar
      </button>
      <Sidebar ref={sidebarRef} type="animation_api" />
    </div>
  );
}

interface SidebarProps {
  onTransitionEnd?: React.TransitionEventHandler<HTMLDivElement>;
  ref: React.Ref<HTMLDivElement>;
  style?: React.CSSProperties;
  type: 'animation_api' | 'max_width' | 'width';
}

function Sidebar({ ref, type, style, onTransitionEnd }: SidebarProps) {
  return (
    <div ref={ref} className={classes({ type })} style={style} onTransitionEnd={onTransitionEnd}>
      {Array.from({ length: 5 }).map((_, idx) => (
        <div key={idx} className="flex gap-2 px-2 py-1">
          <div className="size-6 bg-neutral-200" />
          <div className="h-6 w-60 bg-neutral-200" />
        </div>
      ))}
    </div>
  );
}

const classes = cva(
  'absolute top-0 right-0 h-full flex-col gap-2 border-l border-l-neutral-200 py-2',
  {
    variants: {
      type: {
        max_width: 'flex overflow-hidden transition-[max-width] duration-250',
        width: 'hidden overflow-hidden transition-[width] duration-250',
        animation_api: 'hidden overflow-hidden',
      },
    },
  },
);
