import { os, type } from '@orpc/server';
import { RPCHandler } from '@orpc/server/message-port';

const get_date = os.handler(() => {
  return new Date();
});

const render = os.input(type<{ canvas: OffscreenCanvas }>()).handler(({ input }) => {
  const ctx = input.canvas.getContext('2d');
  if (!ctx) throw new Error('Failed to get 2D context from OffscreenCanvas');

  ctx.fillStyle = 'green';

  const circle = new AnimatedCircle(ctx);
  circle.bound_animate();
});

class AnimatedCircle {
  ctx: OffscreenCanvasRenderingContext2D;
  x: number;
  y: number;
  radius: number;
  max_radius: number;
  grow: boolean;
  bound_animate: () => void;

  constructor(ctx: OffscreenCanvasRenderingContext2D) {
    this.ctx = ctx;
    this.x = ctx.canvas.width / 2;
    this.y = ctx.canvas.height / 2;
    this.radius = 24;
    this.max_radius = 36;
    this.grow = true;
    this.bound_animate = this.animate.bind(this);
  }

  draw() {
    this.ctx.beginPath();
    this.ctx.arc(this.x, this.y, this.radius, 0, 2 * Math.PI, false);
    this.ctx.fill();
  }

  animate() {
    if (this.radius === this.max_radius || this.radius === 0) {
      this.grow = !this.grow;
    }

    this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
    this.draw();
    this.radius = this.grow ? this.radius + 1 : this.radius - 1;
    requestAnimationFrame(this.bound_animate);
  }
}

export const router = {
  get_date,
  render,
};

const handler = new RPCHandler(router, {
  experimental_transfer: () => {
    return [];
  },
});

handler.upgrade(globalThis as any, {
  context: {},
});
