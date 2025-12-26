import { faker } from '@faker-js/faker';
import { createFileRoute } from '@tanstack/react-router';

export const Route = createFileRoute('/list-virt-fixed-height')({
  validateSearch: (search: Record<string, unknown>) => {
    return {
      variant: (search.variant as 'content-visibility' | 'default') ?? 'default',
    };
  },
  component: FixedHeight,
});

const titles = Array.from({ length: 10_000 }, (_, index) => {
  faker.seed(index);
  return faker.book.title();
});

function FixedHeight() {
  const { variant } = Route.useSearch();

  return (
    <div className="px-2 py-1" style={{ scrollbarGutter: 'stable' }}>
      <div className="grid grid-cols-6 gap-2">
        {titles.map((title, index) => (
          <div
            key={index}
            style={{ ...(variant === 'content-visibility' && { contentVisibility: 'auto' }) }}>
            <div className="h-40 w-full bg-neutral-200" />
            <div className="leading-tight">{title}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
