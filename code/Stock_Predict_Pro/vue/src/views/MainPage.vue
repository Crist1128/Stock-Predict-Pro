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
      <input id="id_search" v-model="searchQuery" placeholder="Search..." />
      <button class="search_button" type="submit">搜索</button>
    </div>
    <div class="hot_stocks">
      <!-- 兴趣列表 -->
      <p>您可能感兴趣的股票：</p>
        <ul>
          <!-- 循环打印股票公司和相关数据 -->
            <li v-for="stock in hotStocks" :key="stock.stock_symbol">
              <span class="stock_symbol">{{ stock.stock_symbol }}</span>
              <span class="company_name">{{ stock.company_name }}</span>
              <span class="latest_close_price">{{ stock.latest_close_price }}</span>
              <span class="change_amount"> {{ stock.change_amount }}%</span>
              <span class="change_percentage">{{ stock.change_percentage }}%</span>
            </li>  
       </ul>
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
    };
  },
  mounted() {
    this.fetchHotStocks();

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
      // axios.get('http://localhost:8000/api/todays_news/')
      //   .then(response => {
      //     this.stockNews = response.data;
      //   })
      //   .catch(error => {
      //     console.error('Error fetching stock news:', error);
      //   });
    },
  },
};
</script>

<style scoped>
@import '../assets/css/index.css';
</style>