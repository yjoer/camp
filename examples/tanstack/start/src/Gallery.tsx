import { createFileRoute, Link } from '@tanstack/react-router';

export const Route = createFileRoute('/')({
  component: RouteComponent,
});

const sections = [
  {
    label: 'Managing State',
    children: [
      { to: '/stale-closures', label: 'Stale Closures' }, //
    ],
  },
  {
    label: 'External Stores',
    children: [
      { to: '/external-stores/redux', label: 'Redux' },
      { to: '/external-stores/redux-toolkit', label: 'Redux Toolkit' },
      { to: '/external-stores/zustand', label: 'Zustand' },
    ],
  },
  {
    label: 'Concurrent Rendering',
    children: [
      { to: '/transition-use-state', label: 'useTransition with useState' },
      { to: '/transition-use-context-selector', label: 'useTransition with useContextSelector' },
      { to: '/transition-use-search', label: 'useTransition with useSearch' },
      { to: '/transition-redux', label: 'useTransition with Redux' },
      { to: '/transition-zustand', label: 'useTransition with Zustand' },
    ],
  },
  {
    label: 'List Virtualization',
    children: [
      { to: '/list-virt-fixed-height', label: 'Fixed Height' },
      { to: '/list-virt-fixed-height?variant=content-visibility', label: 'Fixed Height with Content Visibility' },
      { to: '/list-virt-dynamic-height', label: 'Dynamic Height with uwrap' },
      { to: '/list-virt-dynamic-height?wrap=canvas-hypertxt', label: 'Dynamic Height with canvas-hypertxt' },
    ],
  },
  {
    label: 'Web Components',
    children: [{ to: '/custom-elements', label: 'Custom Elements' }],
  },
  {
    label: 'Video and Audio',
    children: [{ to: '/media-source-extensions', label: 'Media Source Extensions' }],
  },
  {
    label: 'Web Workers',
    children: [{ to: '/worker-offscreen-canvas', label: 'Offscreen Canvas' }],
  },
  {
    label: 'WebGL',
    children: [
      { to: '/webgl/triangle', label: 'Triangle' },
      { to: '/webgl/rectangle', label: 'Rectangle' },
      { to: '/webgl/multiple-rectangles', label: 'Multiple Rectangles' },
    ],
  },
];

const ui_sections = [
  {
    label: 'Animations',
    children: [
      { to: '/ui/animations/sidebar', label: 'Sidebar' }, //
    ],
  },
];

function RouteComponent() {
  return (
    <div className="mx-2 my-1 inline-grid gap-4 md:grid-cols-2">
      <div className="flex flex-col gap-4">
        {sections.map((section, idx) => {
          return (
            <div key={idx}>
              <span className="bg-[#ffdd00]">{section.label}</span>
              {section?.children?.map((child, idx) => {
                return (
                  <Link key={idx} className="block" to={child.to}>
                    {child.label}
                  </Link>
                );
              })}
            </div>
          );
        })}
      </div>
      <div className="flex flex-col gap-4">
        {ui_sections.map((section, idx) => {
          return (
            <div key={idx} className="flex flex-col">
              <div>
                <span className="bg-[#ffa500]">{section.label}</span>
              </div>
              {section?.children?.map((child, idx) => {
                return (
                  <Link key={idx} to={child.to}>
                    {child.label}
                  </Link>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}
