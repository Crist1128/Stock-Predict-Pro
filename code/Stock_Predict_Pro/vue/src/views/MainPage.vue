<!-- // YourComponent.vue -->
<template>
  <div>
    <!-- 导航栏 -->
    <div class="guidance_table">
      <p class="guidance">股市导航>></p>
      <button class="market">市场指数</button>
      <button class="market1">A股</button>
      <button class="market2">美股</button>
    </div>
    <!-- 搜索栏 -->
    <div class="search_table">
      <input id="id_search" v-model="searchQuery" @input="search" placeholder="请输入您想搜索的内容（股票代码或股票名）" @focus="show" />
      <ul v-show="visible" @click="hide" class="search_result_select">
        <li v-for="result in searchResults" :key="result.stock_symbol">
          <div class="search_link1" v-if="result.type === 'stock'" @click="onResultClickStock(result)">
            {{ result.company_name }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ result.stock_symbol
            }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ result.market }}
          </div>
          <div class="search_link2" v-else-if="result.type === 'index'" @click="onResultClickIndex(result)">
            {{ result.index_name }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ result.index_code
            }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ result.market }}
          </div>
        </li>
      </ul>
    </div>
    <div class="hot_stocks">
      <!-- 兴趣列表 -->
      <p>您可能感兴趣的股票：</p>
      <ul>
        <li class="table_header">
          <span class="stock_symbol">股票代码</span>
          <span class="company_name">公司名称</span>
          <span class="latest_close_price">最新收盘价</span>
          <span class="change_amount">涨跌额</span>
          <span class="change_percentage">涨跌幅</span>
        </li>
        <!-- 循环打印股票公司和相关数据 -->
        <li class="table_main" v-for="(stock, index) in visibleStocks" :key="stock.stock_symbol" @click="goHotStock(index)">
          <span class="stock_symbol">{{ stock.stock_symbol }}</span>
          <span class="company_name">{{ stock.company_name }}</span>
          <span class="latest_close_price">{{ stock.latest_close_price.toFixed(2) }}</span>
          <span :class="{ 'font_color_change': stock.change_amount < 0 }" class="change_amount"> {{
            stock.change_amount.toFixed(2) }}%</span>
          <span :class="{ 'negative_change': stock.change_percentage < 0 }" class="change_percentage">{{
            stock.change_percentage.toFixed(2) }}%</span>
        </li>
      </ul>
      <button class="show_more" @click="showMoreStocks">+</button>
    </div>
    <div class="stock_news">
      <!-- 股票新闻 -->
      <p>新闻推荐：</p>
      <ul>
      </ul>
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
      hotStocks: [],
      displayedStocks: 5,
      visible: false,
      i: 0,
    };
  },
  mounted() {
    this.fetchHotStocks();
    document.addEventListener('click', this.hide);
  },
  beforeUnmount() {
    document.removeEventListener('click', this.hideContainer);
  },
  computed: {
    visibleStocks() {
      return this.hotStocks.slice(0, this.displayedStocks);
    },
  },
  methods: {
    // 获取热门股票
    fetchHotStocks() {
      axios.get('http://localhost:8000/api/hot_stocks/')
        .then(response => {
          this.hotStocks = response.data;
        })
        .catch(error => {
          console.error('Error fetching hot stocks:', error);
        });
    },
    // 获取搜索信息
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
    showMoreStocks() {
      this.displayedStocks += 4;
    },
    show() {
      this.visible = true;
    },
    hide(event) {
      // 如果点击的地方不是输入框，则隐藏容器
      if (!event.target.matches('input')) {
        this.visible = false;
      }
    },
    // 实现股票页跳转
    onResultClickStock(result) {
      this.$router.push({ name: 'stockpage', params: { name: result.company_name, id: result.stock_symbol, type: result.market } })
    },
    // 实现指数页跳转
    onResultClickIndex(result) {
      this.$router.push({ name: 'stockpage', params: { name: result.index_name, id: result.index_code, type: result.market } })
    },
    // 热门股票跳转
    goHotStock(index){
      this.$router.push({ name: 'stockpage', params: { name: this.hotStocks[index].company_name, id: this.hotStocks[index].stock_symbol.toLowerCase(), type : "A股"} })
    }
  },
};
</script>

<style scoped>
@import '../assets/css/index.css';
</style>