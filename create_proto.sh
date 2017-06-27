#!/usr/bin/env sh

protoc -I=./src/proto --cpp_out=./include/proto ./src/proto/rdf.proto