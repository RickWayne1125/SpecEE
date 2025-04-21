#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  100
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    std::vector<std::vector<llama_token_data>> dists;

    struct common_sampler * smpl = nullptr;
};
/**
 * This the arbitrary data which will be passed to each callback.
 * Later on we can for example add operation or tensor name filter from the CLI arg, or a file descriptor to dump the tensor.
 */
struct callback_data {
    std::vector<uint8_t> data;
};

#include "log.h"
static void ggml_print_tensor(uint8_t * data, ggml_type type, const int64_t * ne, const size_t * nb, int64_t n) {
    GGML_ASSERT(n > 0);
    float sum = 0;
    for (int64_t i3 = 0; i3 < ne[3]; i3++) {
        LOG("                                     [\n");
        for (int64_t i2 = 0; i2 < ne[2]; i2++) {
            if (i2 == n && ne[2] > 2*n) {
                LOG("                                      ..., \n");
                i2 = ne[2] - n;
            }
            LOG("                                      [\n");
            for (int64_t i1 = 0; i1 < ne[1]; i1++) {
                if (i1 == n && ne[1] > 2*n) {
                    LOG("                                       ..., \n");
                    i1 = ne[1] - n;
                }
                LOG("                                       [");
                for (int64_t i0 = 0; i0 < ne[0]; i0++) {
                    if (i0 == n && ne[0] > 2*n) {
                        LOG("..., ");
                        i0 = ne[0] - n;
                    }
                    size_t i = i3 * nb[3] + i2 * nb[2] + i1 * nb[1] + i0 * nb[0];
                    float v;
                    if (type == GGML_TYPE_F16) {
                        v = ggml_fp16_to_fp32(*(ggml_fp16_t *) &data[i]);
                    } else if (type == GGML_TYPE_F32) {
                        v = *(float *) &data[i];
                    } else if (type == GGML_TYPE_I32) {
                        v = (float) *(int32_t *) &data[i];
                    } else if (type == GGML_TYPE_I16) {
                        v = (float) *(int16_t *) &data[i];
                    } else if (type == GGML_TYPE_I8) {
                        v = (float) *(int8_t *) &data[i];
                    } else {
                        GGML_ABORT("fatal error");
                    }
                    LOG("%12.4f", v);
                    sum += v;
                    if (i0 < ne[0] - 1) LOG(", ");
                }
                LOG("],\n");
            }
            LOG("                                      ],\n");
        }
        LOG("                                     ]\n");
        LOG("                                     sum = %f\n", sum);
    }
}

static std::string ggml_ne_string(const ggml_tensor * t) {
    std::string str;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        str += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) {
            str += ", ";
        }
    }
    return str;
}

static bool ggml_debug(struct ggml_tensor * t, bool ask, void * user_data) {
    bool is_debug = true;
    bool print_tensor = true;
    auto * cb_data = (callback_data *) user_data;

    const struct ggml_tensor * src0 = t->src[0];
    const struct ggml_tensor * src1 = t->src[1];

    if (ask && is_debug) {
        return true; // Always retrieve data
    }

    char src1_str[128] = {0};
    if (src1) {
        snprintf(src1_str, sizeof(src1_str), "%s{%s}", src1->name, ggml_ne_string(src1).c_str());
    }
    const char* prefix = "pred-";
    // 检查 name 是否以 pred- 开头
    bool is_pred = strncmp(t->name, "pred-", strlen("pred-")) == 0 ? true : false;
    long i_layer=0;
    if (is_pred) {
        // 确定第几层il
        i_layer = strtol(t->name + strlen("pred-") , NULL, 10);
        if(print_tensor){
            LOG("callback: pred is need layer %ld \n",i_layer);
        }
    } 
    if(ask && !is_debug){
        return is_pred;
    }
    if(is_debug){
        if(src0==nullptr){
            LOG("%s: %24s = (%s)", __func__,
                t->name, ggml_type_name(t->type),ggml_op_desc(t));
        }else{
            LOG("%s: %24s = (%s) %10s(%s{%s}, %s}) = {%s}\n", __func__,
                t->name, ggml_type_name(t->type), ggml_op_desc(t),
                src0->name, ggml_ne_string(src0).c_str(),
                src1 ? src1_str : "",
                ggml_ne_string(t).c_str());
        }
    }

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (!ggml_is_quantized(t->type)) {
        uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();

        float pred;
        ggml_type tensor_type = t->type;
        if (tensor_type == GGML_TYPE_F16) {
            pred = ggml_fp16_to_fp32(*(ggml_fp16_t *) & data[0]);
        } else if (tensor_type ==  GGML_TYPE_F32) {
            pred = *(float *) & data[0];
        } else if (tensor_type == GGML_TYPE_I32) {
            pred = (float) *(int32_t *) & data[0];
        } else if (tensor_type ==  GGML_TYPE_I16) {
            pred = (float) *(int16_t *) &data[0];
        } else if (tensor_type == GGML_TYPE_I8) {
            pred = (float) *(int8_t *) &data[0];
        } else {
            GGML_ABORT("fatal error");
        }
        if(print_tensor){
            LOG("first element %12.4f\n", pred);
            ggml_print_tensor(data, t->type, t->ne, t->nb, 3);
        }


        bool flag = true;

        // 生成一个随机数 random_number
        // std::mt19937 gen(std::random_device{}());
        // std::uniform_int_distribution<> dis(0, 99);
        // int random_number = dis(gen);
    
        // if(random_number > 90 && i_layer >= 26 && is_pred){
        if(pred>0.5 && is_pred){
            flag = false;
            if(print_tensor){
                LOG("pred > 0.5, try to exit at layer %ld\n", i_layer);
            }
        }
        return flag;
    }


    return true;
}

static bool ggml_pred_callback(struct ggml_tensor * t, bool ask, void * user_data) {

    if(ask){
        // 检查 name 是否以 pred- 开头
        return strncmp(t->name, "pred-", 5) == 0 ? true : false;
    }

    auto * cb_data = (callback_data *) user_data;

    // copy the data from the GPU memory if needed
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);

    if (!is_host) {
        auto n_bytes = ggml_nbytes(t);
        cb_data->data.resize(n_bytes);
        ggml_backend_tensor_get(t, cb_data->data.data(), 0, n_bytes);
    }

    if (ggml_is_quantized(t->type)) {
        GGML_ABORT("fatal error: pred is is quantized");
    }
    uint8_t * data = is_host ? (uint8_t *) t->data : cb_data->data.data();

    float pred;
    ggml_type tensor_type = t->type;
    if (tensor_type == GGML_TYPE_F16) {
        pred = ggml_fp16_to_fp32(*(ggml_fp16_t *) & data[0]);
    } else if (tensor_type ==  GGML_TYPE_F32) {
        pred = *(float *) & data[0];
    } else if (tensor_type == GGML_TYPE_I32) {
        pred = (float) *(int32_t *) & data[0];
    } else if (tensor_type ==  GGML_TYPE_I16) {
        pred = (float) *(int16_t *) &data[0];
    } else if (tensor_type == GGML_TYPE_I8) {
        pred = (float) *(int8_t *) &data[0];
    } else {
        GGML_ABORT("fatal error");
    }

    return pred>0.5?false:true;
}

static bool ggml_is_pred_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    // 检查 name 是否以 pred- 开头
    if(ask){
        return strncmp(t->name, "pred-", 5) == 0 ? true : false;
    }
    return false;
}


int main(int argc, char ** argv) {
    common_params params;
    bool debug_log;
    // needed to get candidate probs even for temp <= 0.0
    params.sparams.n_probs = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    common_init();

    if (params.model_draft.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_split  = params.p_split;

    std::default_random_engine rng(params.sparams.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sparams.seed);
    std::uniform_real_distribution<> u_dist;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_params params_tgt = params;
    if(params.verbosity){
        params_tgt.cb_eval = ggml_debug;
        debug_log = true;
    }else{
        params_tgt.cb_eval = ggml_is_pred_callback;
        debug_log = false;
    }

    callback_data     cb_data;

    params_tgt.cb_eval_user_data = &cb_data;
    common_init_result llama_init_tgt = common_init_from_params(params_tgt);

    model_tgt = llama_init_tgt.model;
    ctx_tgt = llama_init_tgt.context;

    // load the draft model
    params.model = params.model_draft;
    params.n_gpu_layers = params.n_gpu_layers_draft;
    if (params.draft_cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.draft_cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.draft_cpuparams_batch.n_threads;
    common_init_result llama_init_dft = common_init_from_params(params);
    model_dft = llama_init_dft.model;
    ctx_dft = llama_init_dft.context;

    const bool vocab_type_tgt = llama_vocab_type(model_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(model_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_add_bos_token(model_tgt) != llama_add_bos_token(model_dft) ||
        llama_add_eos_token(model_tgt) != llama_add_eos_token(model_dft) ||
        llama_token_bos(model_tgt) != llama_token_bos(model_dft) ||
        llama_token_eos(model_tgt) != llama_token_eos(model_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_n_vocab(model_tgt);
        const int n_vocab_dft = llama_n_vocab(model_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_n_vocab(model_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_token_get_text(model_tgt, i);
            const char * token_text_dft = llama_token_get_text(model_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }


    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1));
    llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input));

    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_n_vocab(model_dft));

    // how many tokens to draft each time
    int n_draft = params.n_draft;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context (reuse the llama_context's sampling instance)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sparams);

    struct llama_sampler * softmax = llama_sampler_init_softmax();

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    for (int s = 0; s < n_seq_dft; ++s) {
        // allocate llama_sampler for each draft sequence
        drafts[s].smpl = common_sampler_init(model_dft, params.sparams);
    }

    llama_batch batch_dft = llama_batch_init(params.n_ctx, 0, 1);
    llama_batch batch_tgt = llama_batch_init(params.n_ctx, 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    while (true) {
        std::set<int> active_seqs = {};

        // print current draft sequences
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            active_seqs.insert(s);
            const auto & tokens = drafts[s].tokens;

            // LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
            if(debug_log){
                LOG_INF("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
            }
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id;
        std::string token_str;

        // loop until we fail to accept a drafted token or we run out of drafted tokens
        while (true) {

            // check if the target token matches any of the drafts
            // for stochastic sampling, attempt to match the token with the drafted tokens
            {
                bool accept = false;
                {
                    // greedy verification

                    // sample from the target model
                    // LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    if(debug_log){
                        LOG_INF("\nsampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    }
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);

                    common_sampler_accept(smpl, token_id, true);

                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            // LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());
                            if(debug_log){
                                LOG_INF("\nthe sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());
                            }
                            s_keep = s;
                            accept = true;
                        } else {
                            drafts[s].active = false;
                        }
                    }
                }

                if (llama_token_is_eog(model_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;

                if (accept) {
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    if(n_predict>=n_draft){
                        break;
                    }
                    continue;
                } else {
                    LOG("%s", token_str.c_str());
                    break;
                }
            }
        }

        {
            if(debug_log){
                LOG_INF("\nthe sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());
            }
            // TODO: simplify
            {
                if(debug_log){
                    LOG_INF("\nkeeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);
                }

                llama_kv_cache_seq_keep(ctx_dft, s_keep);
                llama_kv_cache_seq_cp  (ctx_dft, s_keep, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_dft, 0);

                llama_kv_cache_seq_rm  (ctx_tgt, s_keep, n_past_tgt, -1);
                llama_kv_cache_seq_keep(ctx_tgt, s_keep);
                llama_kv_cache_seq_cp  (ctx_tgt, s_keep, 0, -1, -1);
                llama_kv_cache_seq_keep(ctx_tgt, 0);
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(token_id);
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            common_batch_clear(batch_dft);
            common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true);

            llama_kv_cache_seq_rm(ctx_dft, 0, n_past_dft, -1);
            // LOG_DBG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
            llama_decode(ctx_dft, batch_dft);

            ++n_past_dft;
        }

        if (n_predict > params.n_predict || has_eos) {
            break;
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl);

                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                    if(debug_log){
                        LOG_INF(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                    }
                }
                // 向target model传递candidate_tokens和n_candidates
                batch_add_candidates(batch_tgt, cur_p->data, std::min(n_seq_dft + 3, (int) cur_p->size));

                std::vector<int> sa(1, s);

                // attempt to split the branch if the probability is high enough


                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const llama_token id = cur_p->data[is].id;

                    const int s = sa[is];

                    common_sampler_accept(drafts[s].smpl, id, true);

                    drafts[s].tokens.push_back(id);
                    // save cur_p.data into drafts[s].dists
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    // add unique drafted tokens to the target batch
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                    // common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                    // add the token to the batch for batched decoding with the draft model
                    drafts[s].i_batch_dft = batch_dft.n_tokens;

                    common_batch_add(batch_dft, id, n_past_cur, { s }, true);

                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            }

            // no sequence is drafting anymore
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            llama_decode(ctx_dft, batch_dft);
            ++n_past_cur;
            ++n_drafted;

            if (batch_tgt.n_tokens >= n_draft) {
                break;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            llama_kv_cache_seq_keep(ctx_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_kv_cache_seq_cp(ctx_tgt, 0, s, -1, -1);
            }

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            llama_decode(ctx_tgt, batch_tgt);
            ++n_past_tgt;
        }

        // the first token is always proposed by the target model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    // TODO: print sampling/grammar timings for all drafts
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_sampler_free(softmax);
    llama_batch_free(batch_dft);

    llama_free(ctx_tgt);
    llama_free_model(model_tgt);

    llama_free(ctx_dft);
    llama_free_model(model_dft);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
