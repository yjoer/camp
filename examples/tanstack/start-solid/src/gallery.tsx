import { createFileRoute, Link } from '@tanstack/solid-router';
import { For } from 'solid-js';

export const Route = createFileRoute('/')({
  component: Gallery,
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
      { to: '/transition-signal', label: 'useTransition with createSignal' }, //
    ],
  },
];

function Gallery() {
  return (
    <div class="mx-2 my-1 flex flex-col gap-4">
      <For each={sections}>
        {(section) => {
          return (
            <div class="flex flex-col">
              <div>
                <span class="bg-[#ffdd00]">{section.label}</span>
              </div>
              <For each={section?.children}>
                {(child) => {
                  return <Link to={child.to}>{child.label}</Link>;
                }}
              </For>
            </div>
          );
        }}
      </For>
    </div>
  );
}
