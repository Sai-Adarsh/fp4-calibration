cmake_minimum_required(VERSION 3.24)

string(RANDOM LENGTH 12 ID)
set(WORK "${TEST_ROOT}/quant-cli-${ID}")
file(MAKE_DIRECTORY "${WORK}/tools")

function(expect_cli status pattern)
    execute_process(
        COMMAND "${QUANT}" ${ARGN}
        RESULT_VARIABLE actual
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        TIMEOUT 30
    )
    if(NOT actual STREQUAL "${status}" OR NOT "${stdout}${stderr}" MATCHES "${pattern}")
        message(FATAL_ERROR "Expected exit ${status}, /${pattern}/. Got ${actual}: ${stdout}${stderr}")
    endif()
endfunction()

function(match_file path pattern)
    file(READ "${path}" contents)
    if(NOT contents MATCHES "${pattern}")
        message(FATAL_ERROR "${path} did not match /${pattern}/")
    endif()
endfunction()

file(WRITE "${WORK}/model with spaces.gguf" "GGUF fake F32 payload with larger bytes")
file(WRITE "${WORK}/calibration.txt" "A student drafted a schedule for the hiking trip.\n")
file(WRITE "${WORK}/evaluation.txt" "The bridge was closed before the morning walk.\n")
file(WRITE "${WORK}/training.txt" "Weekend hikers chose a route around the river.\n")
file(WRITE "${WORK}/empty.txt" "")
file(WRITE "${WORK}/example-00001-of-00002.gguf" "GGUF fake shard")
file(COPY_FILE "${WORK}/calibration.txt" "${WORK}/copy.txt")

expect_cli(0 "byte_model_charged_transitions=147.*FP32 bytes=262144.*E2M1 bytes=40960.*INT4 bytes=40960")
expect_cli(0 "FP32 bytes=262144" "${WORK}/training.txt" "${WORK}/calibration.txt" "${WORK}/evaluation.txt")
expect_cli(1 "usage:" bad)
expect_cli(1 "cannot read corpus" "${WORK}/missing.txt" "${WORK}/calibration.txt" "${WORK}/evaluation.txt")
expect_cli(1 "corpus must contain at least two bytes" "${WORK}/empty.txt" "${WORK}/calibration.txt" "${WORK}/evaluation.txt")
expect_cli(1 "use distinct" "${WORK}/calibration.txt" "${WORK}/calibration.txt" "${WORK}/evaluation.txt")

set(GGUF gguf "${WORK}/tools" "${WORK}/model with spaces.gguf"
         "${WORK}/calibration.txt" "${WORK}/evaluation.txt")
expect_cli(1 "usage:" gguf)
expect_cli(1 "unsupported GGUF type" ${GGUF} "${WORK}/bad-format" Q8)
expect_cli(1 "expected a nonnegative integer" ${GGUF} "${WORK}/bad-layers" Q4_K_M -1)
expect_cli(1 "minimum context" ${GGUF} "${WORK}/bad-context" Q4_K_M 0 31)
expect_cli(1 "budget must be finite and nonnegative" ${GGUF} "${WORK}/bad-budget" Q4_K_M 0 128 NaN)
expect_cli(1 "single-file GGUF" gguf "${WORK}/tools" "${WORK}/example-00001-of-00002.gguf"
           "${WORK}/calibration.txt" "${WORK}/evaluation.txt" "${WORK}/split")
expect_cli(1 "not a regular file" gguf "${WORK}/tools" "${WORK}/missing.gguf"
           "${WORK}/calibration.txt" "${WORK}/evaluation.txt" "${WORK}/missing")
expect_cli(1 "texts must be nonempty" gguf "${WORK}/tools" "${WORK}/model with spaces.gguf"
           "${WORK}/calibration.txt" "${WORK}/empty.txt" "${WORK}/empty")
expect_cli(1 "same SHA-256" gguf "${WORK}/tools" "${WORK}/model with spaces.gguf"
           "${WORK}/calibration.txt" "${WORK}/copy.txt" "${WORK}/duplicates")
expect_cli(1 "missing executable" ${GGUF} "${WORK}/no-tools")
if(EXISTS "${WORK}/duplicates")
    message(FATAL_ERROR "Duplicate corpora left a result directory")
endif()

foreach(tool llama-imatrix llama-quantize llama-perplexity llama-bench)
    file(COPY_FILE "${FAKE_LLAMA}" "${WORK}/tools/${tool}")
    file(CHMOD "${WORK}/tools/${tool}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
endforeach()

expect_cli(0 "/summary.csv" ${GGUF} "${WORK}/passed" Q4_K_M 0 128 10)
file(SHA256 "${WORK}/calibration.txt" calibration_hash)
file(SHA256 "${WORK}/evaluation.txt" evaluation_hash)
match_file("${WORK}/passed/manifest.csv" "calibration,.*${calibration_hash}")
match_file("${WORK}/passed/manifest.csv" "evaluation,.*${evaluation_hash}")
match_file("${WORK}/passed/summary.csv" "Q4_K_M,[0-9]+,105,1[.]75,5,5,0[.]0487")
match_file("${WORK}/passed/quality-gate.csv" "10,5,pass")
match_file("${WORK}/passed/commands.txt" "model with spaces.gguf")
match_file("${WORK}/passed/quantized.bench.csv" "0,128,50")

expect_cli(3 "perplexity increase 5% exceeds 1% budget" ${GGUF} "${WORK}/failed" MXFP4_MOE 0 128 1)
match_file("${WORK}/failed/summary.csv" "MXFP4_MOE,")
match_file("${WORK}/failed/quality-gate.csv" "1,5,fail")
expect_cli(0 "/summary.csv" ${GGUF} "${WORK}/unbounded" Q4_K_M 0 128)
if(EXISTS "${WORK}/unbounded/quality-gate.csv")
    message(FATAL_ERROR "Unexpected gate without a budget")
endif()

expect_cli(1 "output directory already exists" ${GGUF} "${WORK}/passed" Q4_K_M 0 128)
set(ENV{QUANT_TEST_MODE} "crash")
expect_cli(1 "llama-perplexity failed" ${GGUF} "${WORK}/crash" Q4_K_M 0 128)
match_file("${WORK}/crash/baseline.ppl.stderr" "simulated subprocess failure")
set(ENV{QUANT_TEST_MODE} "missing_ppl")
expect_cli(1 "missing final perplexity" ${GGUF} "${WORK}/missing-ppl" Q4_K_M 0 128)
foreach(role calibration quantized)
    set(ENV{QUANT_TEST_MODE} "mutate_${role}")
    set(result "${WORK}/mutated-${role}")
    expect_cli(1 "artifact changed during run: ${role}" ${GGUF} "${result}" Q4_K_M 0 128 10)
    if(EXISTS "${result}/summary.csv" OR EXISTS "${result}/quality-gate.csv")
        message(FATAL_ERROR "Changed ${role} produced certified measurements")
    endif()
    match_file("${result}/manifest.csv" "${role},")
    match_file("${result}/commands.txt" "llama-bench")
endforeach()
unset(ENV{QUANT_TEST_MODE})
file(REMOVE_RECURSE "${WORK}")
