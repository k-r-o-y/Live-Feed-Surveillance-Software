// detector.worker.js
/* eslint-disable no-undef */
let model = null;
let ready = false;

// Load TFJS + COCO-SSD inside the worker
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js");
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js");

// Use CPU backend in worker (safe). WebGL isn't available in workers.
(async () => {
  try {
    await tf.setBackend("cpu");
    await tf.ready();
    model = await cocoSsd.load();
    ready = true;
    postMessage({ type: "ready" });
  } catch (err) {
    postMessage({ type: "error", message: String(err?.message || err) });
  }
})();

// We draw ImageData into an OffscreenCanvas and run model.detect(canvas)
let osc = null;
let oscCtx = null;

onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === "detect") {
    if (!ready || !model) return;

    const { width, height, threshold } = msg;
    const buf = msg.buffer; // ArrayBuffer transferred from main

    try {
      // Rebuild ImageData from transferred buffer
      const u8 = new Uint8ClampedArray(buf);
      const imageData = new ImageData(u8, width, height);

      // Ensure OffscreenCanvas matches
      if (!osc || osc.width !== width || osc.height !== height) {
        osc = new OffscreenCanvas(width, height);
        oscCtx = osc.getContext("2d", { willReadFrequently: false });
      }

      oscCtx.putImageData(imageData, 0, 0);

      const preds = await model.detect(osc);

      const people = preds
        .filter((p) => p.class === "person" && p.score >= threshold)
        .map((p) => {
          const [x, y, w, h] = p.bbox;
          return { x, y, w, h, score: p.score };
        });

      postMessage({ type: "result", width, height, people });
    } catch (err) {
      postMessage({ type: "error", message: String(err?.message || err) });
    }
  }
};
