import * as stylex from '@stylexjs/stylex';

export const button_styles = stylex.create({
  base: {
    paddingBlock: 4,
    paddingInline: 12,
    cursor: 'pointer',
    outline: 'none',
    backgroundColor: 'rgba(0, 0, 0, 0.04)',
    borderRadius: 4,
    scale: {
      ':active': 0.95,
    },
    transition: 'scale 0.1s cubic-bezier(0.4, 0, 0.2, 1)',
  },
});
