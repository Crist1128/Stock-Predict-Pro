<template>
    <!-- 导航栏 -->
    <div class="guidance_table">
        <p class="guidance">股市导航>></p>
        <button class="market">市场指数</button>
        <button class="market1">A股</button>
        <button class="market2">美股</button>
    </div>
    <!-- 搜索栏 -->
    <div class="container">
        <div class="search_table">
            <input id="id_search" v-model="searchQuery" @input="search" placeholder="请输入您想搜索的内容（股票代码或股票名）" @focus="show" />
              <ul v-show="visible" @click="hide" class="search_result_select">
                <li v-for="result in searchResults" :key="result.stock_symbol">
                  <div class="search_link1" v-if="result.type === 'stock'" @click="onResultClickStock(result)"> 
                    {{ result.company_name }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ result.stock_symbol }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ result.market }}
                  </div>
                  <div class="search_link2" v-else-if="result.type === 'index'" @click="onResultClickIndex(result)">
                    {{ result.index_name }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ result.index_code }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ result.market }}
                  </div>
                </li>
              </ul>
          </div>
    </div>
    <!-- 股票信息 -->
    <div class="mainBox">
        <!-- 股票标识 -->
        <div class="stockName">
        <a href="../../../mainpage">>&nbsp;首页&nbsp;&nbsp;</a>
        <span>{{ stock_information.symbol}}</span>
        <p>{{ stock_information.stock_name }}</p>
        <!-- <p v-if="stock_information.current.price">{{ stock_information.current_price.price }}</p>
        <p v-if="stock_information.current_price">{{ stock_information.current_price.timestamp }}</p> -->
        </div>


        <!-- 预测图 ：交给林显俊-->
        <div class="stockPredictPhoto"></div>


        <!-- 市场信息 -->
        <div class="marketInformation">
            <ul>
                <li><span>昨日收盘价</span><span class="right">{{ stock_information.previous_close }}</span></li>
                <li v-if="stock_information.price_range"><span>当日价格范围</span><span class="right">{{ stock_information.price_range.low }}-{{ stock_information.price_range.high }}</span></li>
                <li><span>年度波幅</span><span class="right">{{ stock_information.year_to_date_return }}</span></li>
                <li><span>市值</span><span class="right">{{ stock_information.market_cap }}</span></li>
                <li><span>平均交易量</span><span class="right">{{ stock_information.average_volume }}</span></li>
            </ul>
        </div>
        <!-- 简介 -->
        <div class="introduction">
            <p>简介</p>
        </div>
    </div>
</template>

<script>
import axios from 'axios';
export default {
    data() {
        return {
            searchQuery: '',
            searchResults: [],
            visible: false,
            stock_id: null,
            stock_information: [],
            stock_information_replace:[],
        };
    },
    mounted() {
        this.getstock();
        document.addEventListener('click', this.hide);
    },
    beforeUnmount() {
        document.removeEventListener('click', this.hide);
    },  
    methods: {
        // 搜索结果接口
        search() {
            if (this.searchQuery.length >= 3) {
                axios.get(`http://localhost:8000/api/search/?query=${this.searchQuery}`)
                    .then(response => {
                        this.searchResults = response.data;
                    })
                    .catch(error => {
                        console.error('Error fetching search results:', error);
                    });
            } else {
                this.searchResults = [];
            }
        },
        getstock() {
            this.stock_id = this.$route.params.id;
            axios.get(`http://localhost:8000/api/stock/${this.stock_id}/`)
                .then(response => {
                    this.stock_information = response.data;
                })
                .catch(error => {
                    console.error('Error fetching search results:', error);
                });
        },
        // 显示搜索选择栏
        show() {
            this.visible = true;
        },
        // 隐藏搜索选择栏
        hide(event) {
            if (!event.target.matches('input') && !event.target.closest('.search_link1') && !event.target.closest('.search_link2')) {
                this.visible = false;
            }
        },
        // 实现股票页跳转
        onResultClickStock(result) {
            this.$router.push({ name: 'stockpage', params: { name: result.company_name, id: result.stock_symbol, type: result.market } })
            this.stock_id = result.stock_symbol;  
            // 获取股票信息
            axios.get(`http://localhost:8000/api/stock/${this.stock_id}/`)
                .then(response => {
                    this.stock_information = response.data;
                })
                .catch(error => {
                    console.error('Error fetching search results:', error);
                });  
        },
        // 实现指数页跳转
        onResultClickIndex(result) {
            this.$router.push({ name: 'stockpage', params: { name: result.index_name, id: result.index_code, type: result.market } })
            this.stock_id = result.index_code;
            // 获取指数信息
            axios.get(`http://localhost:8000/api/stock/${this.stock_id}/`)
                .then(response => {
                    this.stock_information = response.data;
                })
                .catch(error => {
                    console.error('Error fetching search results:', error);
                });
        },
    },
};
</script>

<style scoped>  
@import '../assets/css/stock.css';
</style>