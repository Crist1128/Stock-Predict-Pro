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
      <input id="id_search" v-model="searchQuery" placeholder="   请输入您想搜索的内容（股票代码）" />
      <!-- <button class="search_button" type="submit">搜索</button> -->
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
          <li class="table_main" v-for="stock in visibleStocks" :key="stock.stock_symbol">
            <span class="stock_symbol">{{ stock.stock_symbol }}</span>
            <span class="company_name">{{ stock.company_name }}</span>
            <span class="latest_close_price">{{ stock.latest_close_price.toFixed(2)  }}</span>
            <span :class="{'font_color_change':stock.change_amount < 0}" class="change_amount"> {{ stock.change_amount.toFixed(2)  }}%</span>
            <span :class="{ 'negative_change': stock.change_percentage < 0 }" class="change_percentage">{{ stock.change_percentage.toFixed(2)  }}%</span>
          </li>  
       </ul>
       <button class="show_more" @click="showMoreStocks">Loading more</button>
    </div>
    <div class="stock_news">
      <!-- 股票新闻 -->
      <p>新闻推荐：</p>
        <ul>
            <!-- 循环打印最新股票新闻 -->
              <!-- <li v-for="news in stockNews" :key="news.stock_symbol">
                
              </li>   -->
         </ul>
    </div>
  </div>
</template>

<script>
import axios from 'axios';


export default {
  data() {
    return {
      hotStocks: [],
      displayedStocks: 5,
    };
  },
  mounted() {
    this.fetchHotStocks();
  },
  computed: {
    visibleStocks() {
      return this.hotStocks.slice(0, this.displayedStocks);
    },
  },
  methods: {
    fetchHotStocks() {
      axios.get('http://localhost:8000/api/hot_stocks/')
        .then(response => {
          this.hotStocks = response.data;
        })
        .catch(error => {
          console.error('Error fetching hot stocks:', error);
        });
    },
    showMoreStocks() {
      this.displayedStocks += 4;
    },
  },
};
</script>

<style scoped>
@import '../assets/css/index.css';
</style>