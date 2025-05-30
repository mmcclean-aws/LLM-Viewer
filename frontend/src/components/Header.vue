<template>
    <div class="title">
        <a href="https://github.com/hahnyuan/LLM-Viewer" target="_blank" class="hover-bold">LLM-Viewer</a>
        v{{ version }}
    </div>
    <div class="header_button">
        |
        <span>Model: </span>
        <select v-model="select_model_id">
            <option v-for="model_id in avaliable_model_ids" :value="model_id">{{ model_id }}</option>
        </select>
        <span> | </span>
        <span>Hardware: </span>
        <select v-model="select_hardware">
            <option v-for="hardware in avaliable_hardwares" :value="hardware">{{ hardware }}</option>
        </select>
    </div>
    <div>
        <span> | </span>
        <span>Server: </span>
        <select v-model="ip_port">
            <option value="127.0.0.1:8000">127.0.0.1</option>
        </select>
    </div>
    <div>
        <span> | </span>
        <span class="hover-bold" @click="is_show_help = ! is_show_help">Help</span>
    </div>
    <div>
        <span> | </span>
        <a href="https://github.com/hahnyuan/LLM-Viewer" target="_blank" class="hover-bold">Github Project</a>
    </div>
    <div>
        <span> | </span>
        <a href="https://arxiv.org/pdf/2402.16363.pdf" target="_blank" class="hover-bold">Paper</a>
    </div>
    <div v-if="is_show_help" class="float-info-window">
        <!-- item -->
        <p>LLM-Viewer is a open-sourced tool to visualize the LLM model and analyze the deployment on hardware devices.</p>
        <p>
            At the center of the page, you can see the graph of the LLM model. Click the node to see the detail of the node.
        </p>
        <p>↑ At the top of the page, you can set the LLM model, hardware devices, and server.
            If you deploy the LLM-Viewer localhost, you can select the localhost server.
        </p>
        <p>
            ← At the left of the page, you can see the configuration pannel. You can set the inference config and optimization config.
        </p>
        <p>
            ↙ The Network-wise Analysis result is demonstrated in the left pannel.
        </p>
        <p>
            We invite you to read our paper <a class="hover-bold" href="https://arxiv.org/pdf/2402.16363.pdf" target="_blank">LLM Inference Unveiled: Survey and Roofline Model Insights</a>.
            In this paper, we provide a comprehensive analysis of the latest advancements in efficient LLM inference using LLM-Viewer. 
            Citation bibtext:
        </p>
        @article{yuan2024llm,<br/>
            &nbsp    title={LLM Inference Unveiled: Survey and Roofline Model Insights},<br/>
            &nbsp    author={Yuan, Zhihang and Shang, Yuzhang and Zhou, Yang and Dong, Zhen and Xue, Chenhao and Wu, Bingzhe and Li, Zhikai and Gu, Qingyi and Lee, Yong Jae and Yan, Yan and others},<br/>
            &nbsp    journal={arXiv preprint arXiv:2402.16363},<br/>
            &nbsp    year={2024}<br/>
        }
    </div>
</template>

<script setup>
import { inject, ref, watch, computed, onMounted } from 'vue';
import axios from 'axios'
const model_id = inject('model_id');
const hardware = inject('hardware');
const global_update_trigger = inject('global_update_trigger');
const ip_port = inject('ip_port');

const avaliable_hardwares = ref([]);
const avaliable_model_ids = ref([]);

const version = ref(llm_viewer_frontend_version)

const is_show_help = ref(false)

function update_avaliable() {
    const url = 'http://' + ip_port.value + '/get_avaliable'
    axios.get(url).then(function (response) {
        console.log(response);
        avaliable_hardwares.value = response.data.avaliable_hardwares
        avaliable_model_ids.value = response.data.avaliable_model_ids

        if (
            !select_hardware.value ||
            !avaliable_hardwares.value.includes(select_hardware.value)
        ) {
            select_hardware.value = avaliable_hardwares.value[0] || '';
        }
    })
        .catch(function (error) {
            console.log("error in get_avaliable");
            console.log(error);
        });
}

onMounted(() => {
    console.log("Header mounted")
    update_avaliable()
})

var select_model_id = ref('meta-llama/Llama-2-7b-hf');
watch(select_model_id, (n) => {
    console.log("select_model_id", n)
    model_id.value = n
    global_update_trigger.value += 1
})

var select_hardware = ref('nvidia_V100');
watch(select_hardware, (n) => {
    console.log("select_hardware", n)
    hardware.value = n
    global_update_trigger.value += 1
})

watch(ip_port, (n) => {
    console.log("ip_port", n)
    update_avaliable()
})


</script>

<style scoped>
.header_button button {
    font-size: 1.0rem;
    margin: 5px;
    padding: 5px;
    border-radius: 5px;
    border: 1px solid #000000;
    /* background-color: #fff; */
    /* color: #000; */
    cursor: pointer;
}

.header_button button:hover {
    color: #fff;
    background-color: #000;
}

.header_button button:active {
    color: #fff;
    background-color: #000;
}

.active {
    color: #fff;
    background-color: #5b5b5b;
}



.title {
    font-size: 18px;
    /* 左对齐 */
    text-align: left;
}

.hover-bold{
    color: inherit;
    /* text-decoration: none; */
}

.hover-bold:hover {
    font-weight: bold;
}


.float-info-window {
    position: absolute;
    top: 80px;
    left: 40%;
    height: auto;
    width: 30%;
    background-color: #f1f1f1ed;
    padding: 20px;
    /* background-color: #fff; */
    /* border: 2px solid #4e4e4e; */
    z-index: 999;
}
</style>