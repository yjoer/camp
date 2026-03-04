// oxlint-disable import/no-default-export
// oxlint-disable no-unreachable
const mod = typeof __resourceQuery === 'string' ? __resourceQuery.slice(1) : 'unknown';
throw new Error(`optional dependency "${mod}" could not be resolved at build time.`);

export default {};
