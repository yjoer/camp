/* eslint-disable import-x/no-extraneous-dependencies */
import { setProjectAnnotations } from '@storybook/react-vite';
import { beforeAll } from 'vitest';

import * as projectAnnotations from './preview';

const annotations = setProjectAnnotations([projectAnnotations]);

beforeAll(annotations.beforeAll);
