#!/bin/bash

# go to crates/prover
cd crates/prover

wasm-pack build --target web --release -- --features parallel

rm -rf ../../web/pkg
cp -r pkg ../../web/pkg
rm -rf pkg

cd ../../web
npm run dev
