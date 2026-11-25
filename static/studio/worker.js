import {Decoder, search} from './engine.js';
let ready;
async function load() {
  const responses = await Promise.all(['model.json', 'weights.bin', 'recipes.json'].map(p => fetch(new URL(`data/${p}`, import.meta.url))));
  if (responses.some(r => !r.ok)) throw new Error('MODEL_LOAD_FAILED');
  const [metadata, buffer, history] = await Promise.all([responses[0].json(), responses[1].arrayBuffer(), responses[2].json()]);
  return {decoder: new Decoder(metadata, buffer), history};
}
self.onmessage = async ({data}) => {
  try {
    ready ||= load();
    const {decoder, history} = await ready;
    const results = search(decoder, history, data.target, data.fabric,
      (done, total) => self.postMessage({type: 'progress', id: data.id, done, total}));
    self.postMessage({type: 'result', id: data.id, results});
  } catch (error) { self.postMessage({type: 'error', id: data.id, message: error.message}); }
};
