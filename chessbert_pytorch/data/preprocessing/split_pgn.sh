#!/bin/bash
src="sample_data/subset.pgn"
games=10

split -l $((18*$games)) -d --additional-suffix=.pgn $src ""
