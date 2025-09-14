import { useEffect, useRef } from 'react';

export const Route = createFileRoute({
  component: MediaSourceExtensions,
});

function MediaSourceExtensions() {
  const ref = useRef<HTMLDivElement>(null!);

  useEffect(() => {
    const video = document.createElement('video');
    video.controls = true;
    ref.current.append(video);

    const ms = new MediaSource();
    video.src = URL.createObjectURL(ms);

    ms.addEventListener('sourceopen', async (e) => {
      URL.revokeObjectURL(video.src);
      const ms = e.target as MediaSource;
      const sb = ms.addSourceBuffer('video/mp4; codecs="avc1.42E01E, mp4a.40.2"');
      sb.addEventListener('updateend', () => {
        if (!sb.updating && ms.readyState === 'open') ms.endOfStream();
      });

      const res = await fetch('https://nickdesaulniers.github.io/netfix/demo/frag_bunny.mp4');
      const buf = await res.arrayBuffer();
      sb.appendBuffer(buf);
    });

    return () => {
      video.remove();
      if (ms.readyState === 'open') ms.endOfStream();
    };
  }, []);

  return <div ref={ref} className="mx-2 my-1" />;
}
