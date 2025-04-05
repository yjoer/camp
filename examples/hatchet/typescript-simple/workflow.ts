import { hatchet } from './hatchet-client.ts';

export type HelloInput = {
  message: string;
};

export const hello = hatchet.task({
  name: 'hello',
  fn: (input: HelloInput) => {
    return {
      transformedMessage: input.message.toLowerCase(),
    };
  },
});
