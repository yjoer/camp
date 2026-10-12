const IQR_MULTIPLIER = 1.5;

export function stats(measurements: number[]) {
	const sorted = measurements.toSorted((a, b) => a - b);
	const lower = percentile(sorted, 25);
	const upper = percentile(sorted, 75);

	const range = upper - lower;
	const lower_fence = lower - IQR_MULTIPLIER * range;
	const upper_fence = upper + IQR_MULTIPLIER * range;

	let n = 0, mean = 0, m2 = 0;
	let max = -Infinity;
	const outliers: number[] = [];
	for (const x of measurements) {
		if (x > max) max = x;
		if (x < lower_fence || x > upper_fence) {
			outliers.push(x);
			continue;
		}

		// welford's algorithm
		n += 1;
		const delta = x - mean;
		mean += delta / n;
		const delta2 = x - mean;
		m2 += delta * delta2;
	}

	const std = Math.sqrt(m2 / n);
	return { n, outliers, max, mean, std, ratio: std / mean };
}

export function percentile(array: number[], p: number) {
	const pos = (array.length - 1) * p / 100;
	const lo = Math.floor(pos);
	const hi = Math.ceil(pos);
	return array[lo] + (pos - lo) * (array[hi] - array[lo]);
};

if (import.meta.vitest) {
	const { describe, expect, test } = import.meta.vitest;

	describe('percentile', () => {
		test('interpolates between ranks', () => {
			expect(percentile([1, 2, 3, 4], 25)).toBe(1.75);
			expect(percentile([1, 2, 3, 4], 50)).toBe(2.5);
		});

		test('returns ends at 0 and 100', () => {
			expect(percentile([1, 2, 3, 4], 0)).toBe(1);
			expect(percentile([1, 2, 3, 4], 100)).toBe(4);
		});
	});

	describe('stats', () => {
		test('excludes outliers from n/mean/std but tracks max', () => {
			const result = stats([10, 10, 10, 10, 1000]);
			expect(result.n).toBe(4);
			expect(result.outliers).toEqual([1000]);
			expect(result.max).toBe(1000);
			expect(result.mean).toBe(10);
			expect(result.std).toBe(0);
			expect(result.ratio).toBe(0);
		});

		test('welford mean and std over inliers', () => {
			const result = stats([1, 2, 3, 4, 5]);
			expect(result.n).toBe(5);
			expect(result.outliers).toEqual([]);
			expect(result.mean).toBe(3);
			expect(result.std).toBeCloseTo(Math.SQRT2);
			expect(result.ratio).toBeCloseTo(Math.SQRT2 / 3);
		});
	});
}
