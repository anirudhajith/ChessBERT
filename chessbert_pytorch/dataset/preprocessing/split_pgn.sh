#!/bin/bash
src="data/lichess_db_standard_rated_2020-12.pgn"
games=4000

split -l $((18*$games)) -d --additional-suffix=.pgn $src ""
