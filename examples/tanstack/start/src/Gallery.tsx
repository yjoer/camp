import { Link } from '@tanstack/react-router';

export const Route = createFileRoute({
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
    label: 'Concurrent Rendering',
    children: [
      { to: '/transition-use-state', label: 'useTransition with useState' },
      { to: '/transition-use-context-selector', label: 'useTransition with useContextSelector' },
      { to: '/transition-use-search', label: 'useTransition with useSearch' },
      { to: '/transition-redux', label: 'useTransition with Redux' },
      { to: '/transition-zustand', label: 'useTransition with Zustand' },
    ],
  },
];

function RouteComponent() {
  return (
    <div className="mx-2 my-1 flex flex-col gap-4">
      {sections.map((section, idx) => {
        return (
          <div key={idx} className="flex flex-col">
            <div>
              <span className="bg-[#ffdd00]">{section.label}</span>
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
  );
}
