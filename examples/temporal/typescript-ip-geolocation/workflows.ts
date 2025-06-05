import * as workflow from '@temporalio/workflow';

import type * as activities from './activities.ts';

const { getIP, getLocation } = workflow.proxyActivities<typeof activities>({
  retry: {
    initialInterval: '1s',
    maximumInterval: '1m',
    backoffCoefficient: 2,
  },
  startToCloseTimeout: '1m',
});

export async function getLocationFromIP(name: string) {
  try {
    const ip = await getIP();

    try {
      const location = await getLocation(ip);
      return `Hello, ${name}. Your IP is ${ip} and your location is ${location}.`;
    } catch {
      throw new workflow.ApplicationFailure('failed to get location');
    }
  } catch {
    throw new workflow.ApplicationFailure('failed to get IP');
  }
}
