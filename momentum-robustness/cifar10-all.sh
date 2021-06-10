#!/bin/bash
# CUDA_VISIBLE_DEVICES=4 PYTHONPATH="../../" bash mnist-dissensus-attack.sh
# ps | grep -ie python | awk '{print $1}' | xargs kill -9 


function grid_search_1 {
    for seed in {0..2}
    do
        for m in 0 0.9 0.99
        do
            for agg in "tm" "cm"
            do
                for attack in "LF" "BF" "IPM" "ALIE"
                do
                    python cifar10-all.py --use-cuda --seed $seed --agg $agg --momentum $m --attack $attack &
                    pids[$!]=$!
                done
            done

            # wait for all pids
            for pid in ${pids[*]}; do
                wait $pid
            done
            unset pids
        done
    done
}

function grid_search_2 {
    for seed in {0..2}
    do
        for m in 0 0.9 0.99
        do
            for agg in "rfa" "cp"
            do
                for attack in "LF" "BF" "IPM" "ALIE"
                do
                    python cifar10-all.py --use-cuda --seed $seed --agg $agg --momentum $m --attack $attack &
                    pids[$!]=$!
                done
            done

            # wait for all pids
            for pid in ${pids[*]}; do
                wait $pid
            done
            unset pids
        done
    done
}

function grid_search_3 {
    for seed in {0..2}
    do
        for m in 0 0.9 0.99
        do
            for attack in "LF" "BF" "IPM" "ALIE"
            do
                python cifar10-all.py --use-cuda --seed $seed --agg "krum" --momentum $m --attack $attack &
                pids[$!]=$!
            done
        done

        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
        unset pids
    done
}

function run_avg {
    for seed in {0..2}
    do
        for m in 0 0.9 0.99
        do
            python cifar10-all-noattack.py --use-cuda --seed $seed --agg "avg" --momentum $m &
            pids[$!]=$!
        done

        # wait for all pids
        for pid in ${pids[*]}; do
            wait $pid
        done
        unset pids
    done
}


PS3='Please enter your choice: '
options=("debug" "cprfa" "tmcm" "krum" "avg" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "debug")
            python cifar10-all-debug.py --use-cuda --seed 0 --attack "IPM" --debug --agg "cp" --momentum 0
            ;;

        "cprfa")
            grid_search_2
            ;;

        "tmcm")
            grid_search_1
            ;;

        "krum")
            grid_search_3
            ;;

        "avg")
            run_avg
            ;;

        "Quit")
            break
            ;;

        *) echo "invalid option $REPLY";;
    esac
done


