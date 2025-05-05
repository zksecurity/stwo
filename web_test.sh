#!/bin/bash

rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli

wasm-pack build --target web --release

rm -rf web/pkg
cp -r pkg web/pkg
rm -rf pkg

cd web
npm run dev
