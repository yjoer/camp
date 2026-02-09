/* eslint-disable better-tailwindcss/no-unknown-classes */
import type { Meta, StoryObj } from '@storybook/react-vite';

const meta = {
  title: 'Components/GradientBorderCard',
  component: GradientBorderCard,
  globals: {
    backgrounds: { value: 'dark' },
  },
} satisfies Meta<typeof GradientBorderCard>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    image: 'https://placehold.co/256x144/222/fff',
  },
};

export const Grid: Story = {
  args: {
    image: '',
  },
  render: () => {
    return (
      <div className="flex flex-wrap gap-4">
        <GradientBorderCard image="https://placehold.co/256x144/e63b60/fff" />
        <GradientBorderCard image="https://placehold.co/256x144/00abff/fff" />
        <GradientBorderCard image="https://placehold.co/256x144/a202ff/fff" />
      </div>
    );
  },
};

interface GradientBorderCardProps {
  image: string;
}

function GradientBorderCard({ image }: GradientBorderCardProps) {
  return (
    <div className="w-64">
      <div className="relative overflow-hidden rounded-xl border border-white/4">
        <img
          alt=""
          className="w-full object-cover duration-150 ease-in hover:scale-105"
          src={image}
        />
        <div
          className="
            mask-clip-[content-box,border-box] pointer-events-none absolute inset-0 rounded-[11px]
            bg-[linear-gradient(180deg,rgb(255_255_255/16%),rgb(255_255_255/2%))]
            mask-[linear-gradient(#fff_0_0),linear-gradient(#fff_0_0)] mask-exclude p-px
          "></div>
      </div>
      <h3 className="pt-1 font-semibold text-white/96">Gradient Border Card</h3>
      <p className="line-clamp-1 text-sm text-white/60">
        A frame of light, a tale untold, soft hues wrap secrets, brave and bold.
      </p>
    </div>
  );
}
