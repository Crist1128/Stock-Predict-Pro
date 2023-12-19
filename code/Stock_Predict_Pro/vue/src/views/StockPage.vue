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
      <input id="id_search" v-model="searchQuery" @input="search" placeholder="请输入您想搜索的内容（股票代码或股票名）"
             @focus="show"/>
      <ul v-show="visible" @click="hide" class="search_result_select">
        <li v-for="result in searchResults" :key="result.stock_symbol">
          <div class="search_link1" v-if="result.type === 'stock'" @click="onResultClickStock(result)">
            {{ result.company_name }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ result.stock_symbol }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{
              result.market
            }}
          </div>
          <div class="search_link2" v-else-if="result.type === 'index'" @click="onResultClickIndex(result)">
            {{ result.index_name }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{ result.index_code }}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{{
              result.market
            }}
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
      <span>{{ stock_information.symbol }}</span>
      <p>{{ stock_information.stock_name }}</p>
      <!-- <p v-if="stock_information.current.price">{{ stock_information.current_price.price }}</p>
      <p v-if="stock_information.current_price">{{ stock_information.current_price.timestamp }}</p> -->

    </div>

    <!-- 预测图 ：交给林显俊-->
    <div>
      <!-- 股票信息显示区域 -->
      <div class="stockInfo">
        <p style="font-weight: bolder">开盘价：{{ stock_information.start_price }} 结束价：{{
            stock_information.end_price
          }}</p>
        <p :style="{ color: stock_information.percentage_change > 0 ? '#ce3b3b' : '#23c023'}"
           style="font-size: 30px; font-weight: bold">
          {{ stock_information.percentage_change > 0 ? '↑' : '↓' }}
          {{ Math.abs(stock_information.percentage_change) }}%
        </p>
      </div>
      <!--  时间范围按钮设置-->
      <div class="time-range-buttons" style="margin-bottom: 10px">
        <button @click="changeTimeRange('1D')" :class="{ active: selectedTimeRange === '1D' }">1D</button>
        <button @click="changeTimeRange('5D')" :class="{ active: selectedTimeRange === '5D' }">5D</button>
        <button @click="changeTimeRange('1M')" :class="{ active: selectedTimeRange === '1M' }">1M</button>
        <button @click="changeTimeRange('6M')" :class="{ active: selectedTimeRange === '6M' }">6M</button>
        <button @click="changeTimeRange('1Y')" :class="{ active: selectedTimeRange === '1Y' }">1Y</button>
      </div>
      <!--  股票走势容器-->
      <div ref="stockChart" class="stock-chart"></div>

      <div style="display: flex; max-width: 65%;justify-content: space-between;">
        <!--  复权按钮设置-->
        <div class="stockAdjustButtons">
          <button @click="changestockAdjust('none')" :class="{active: stockAdjust === 'none' }">不复权</button>
          <button @click="changestockAdjust('qfq')" :class="{active: stockAdjust === 'qfq'}">前复权</button>
          <button @click="changestockAdjust('hfq')" :class="{active: stockAdjust === 'hfq'}">后复权</button>
        </div>
        <!--  预测天数按钮-->
        <div class="prediction-container">
          <div class="prediction-input">
            <input v-model="predictionDays" type="number" placeholder="预测天数" :min="1"/>
          </div>
          <button @click="predict()" class="predict-button">预测</button>
        </div>
      </div>
    </div>


    <!-- 市场信息 -->
    <div class="marketInformation">
      <ul>
        <li><span>昨日收盘价</span><span class="right">{{ stock_information.previous_close }}</span></li>
        <li v-if="stock_information.price_range"><span>当日价格范围</span><span
            class="right">{{ stock_information.price_range.low }}-{{ stock_information.price_range.high }}</span></li>
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
import * as echarts from 'echarts';

export default {
  data() {
    return {
      searchQuery: '',
      searchResults: [],
      visible: false,
      stock_id: null,
      stock_information: [],
      stock_information_replace: [],

      selectedTimeRange: '1D', // 默认选中1天
      stockCode: 'sz000001', // 默认股票代码
      stockAdjust: 'none',  // 默认不复权
      predictDay: 1, // 默认预测1天

    };
  },
  mounted() {
    this.getstock();
    document.addEventListener('click', this.hide);

    // 在页面加载后初始化股票走势图
    this.fetchStockData();
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
      this.$router.push({
        name: 'stockpage',
        params: {name: result.company_name, id: result.stock_symbol, type: result.market}
      })
      this.stock_id = result.stock_symbol;
      // 获取股票信息
      axios.get(`http://localhost:8000/api/stock/${this.stock_id}/`)
          .then(response => {
            this.stock_information = response.data;
          })
          .catch(error => {
            console.error('Error fetching search results:', error);
          });

      // 切换stockpage页的股票代码
      this.stockCode = result.stock_symbol;
      this.fetchStockData();
    },
    // 实现指数页跳转
    onResultClickIndex(result) {
      this.$router.push({
        name: 'stockpage',
        params: {name: result.index_name, id: result.index_code, type: result.market}
      })
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

    // 选择时间范围
    changeTimeRange(timeRange) {
      this.selectedTimeRange = timeRange;
      // 更新数据
      this.fetchStockData();
    },
    // 选择复权
    changestockAdjust(Adjust) {
      this.stockAdjust = Adjust;

      this.fetchStockData();
    },
    // 股票走势
    fetchStockData() {
      // 获取股票数据API
      const apiUrl = `http://127.0.0.1:8000/api/stock/${this.stockCode}/price_chart/?time_range=${this.selectedTimeRange}&adjust=${this.stockAdjust}`;

      axios.get(apiUrl)
          .then(response => {
            // 获取到实时股票数据
            const stockData = response.data;

            // 更新股票信息数据
            this.stock_information.start_price = stockData.start_price;
            this.stock_information.end_price = stockData.end_price;
            this.stock_information.percentage_change = stockData.percentage_change;

            // 在页面中找到股票走势图的容器
            const chartContainer = this.$refs.stockChart;

            // 初始化 echarts 实例
            const chart = echarts.init(chartContainer);

            // 处理股票数据，提取时间、价格和交易量
            const timeList = stockData.price_data.map(item => item.time);
            const priceList = stockData.price_data.map(item => item.price);
            const volumeList = stockData.price_data.map(item => item.volume);

            // 计算 Y 轴的最小值，稍微小于最低价格
            const minPrice = Math.min(...priceList);
            const maxPrice = Math.max(...priceList);
            const adjustedMinPrice = minPrice * 0.998;
            const adjustedMaxPrice = maxPrice * 1.008;

            // 根据涨跌率的正负决定线的颜色
            const lineColor = stockData.percentage_change > 0 ? '#ce3b3b' : '#23c023';

            // 配置 echarts 图表选项
            const option = {
              title: {
                text: '股票价格走势',
              },
              tooltip: {
                trigger: 'axis',
                axisPointer: {
                  type: 'cross',
                },
                formatter: function (params) {
                  const dataIndex = params[0].dataIndex;
                  const time = timeList[dataIndex];
                  const price = priceList[dataIndex];
                  const volume = volumeList[dataIndex];
                  return `时间：${time}<br>价格：${price}<br>交易量：${volume}`;
                },
              },
              xAxis: {
                data: timeList,
              },
              yAxis:
                  {
                    type: 'value',
                    name: 'Stock Price',
                    position: 'left',
                    min: adjustedMinPrice, // 设置 Y 轴的最小值
                    max: adjustedMaxPrice,
                  },
              series: [
                {
                  name: 'Stock Price',
                  type: 'line',
                  data: priceList,
                  lineStyle: {
                    color: lineColor, // 设置线的颜色
                  },
                },
              ],
            };

            // 使用配置项设置图表
            chart.setOption(option);
          })
          .catch(error => {
            console.error('Error fetching stock data:', error);
          });
    },

    // 股票预测
    predict() {
      // 构造请求的数据
      const requestData = {
        fq_type: this.stockAdjust,
        predict_days: this.predictDay,
      };

      // 发送POST请求
      axios.post(`http://127.0.0.1:8000/api/stock/${this.stockCode}/predict_daily_close/`, requestData)
          .then(response => {
            // 处理预测结果
            const predictionData = response.data;

            // 获取图表实例
            const chart = this.$refs.stockChart;

            // 处理预测数据，提取时间和价格
            // const predictedTimeList = predictionData.predictions.map(item => item.time);
            const predictedPriceList = predictionData.predictions.map(item => item.price);

            // 配置新的系列（预测数据）
            const predictionSeries = {
              name: 'Prediction',
              type: 'line',
              data: predictedPriceList,
              lineStyle: {
                color: '#953ac9', // 设置预测数据的线的颜色
              },
            };

            // 获取当前图表的配置
            const option = chart.getOption();

            // 添加新的系列到配置中
            option.series.push(predictionSeries);

            // 更新图表配置
            chart.setOption(option);

          })
          .catch(error => {
            console.error('Error predicting stock:', error);
          });
    },
  },
};
</script>

<style scoped>
@import '../assets/css/stock.css';


.stock-chart {
  height: 400px;
  width: 70%;
}

.time-range-buttons {
  display: flex;
  justify-content: space-around; /* 将子元素在主轴上均匀分布 */
  gap: 10px;
  margin-bottom: 2px;
  max-width: 60%; /* 设置容器的最大宽度，根据需要调整 */
}

.time-range-buttons button {
  background: none;
  border: none;
  padding: 0;
  margin: 0;
  font-size: 16px; /* 根据需要调整字体大小 */
  cursor: pointer;
}

.time-range-buttons button.active {
  color: #3c7dcf;
  font-weight: bold;
}

/* stockAdjust按钮设置样式 */
.stockAdjustButtons {
  display: flex;
  gap: 10px;
}

.stockAdjustButtons button {
  background-color: #23c023;
  color: #fff;
  border: none;
  padding: 8px 16px;
  cursor: pointer;
  border-radius: 4px;
  margin-bottom: 5px;
}

.stockAdjustButtons button.active {
  background-color: #239623;
  margin-bottom: 5px;
}

/* stockInfoContainer样式 */
.stockInfo {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-top: 5px;
  margin-bottom: 3px;
}

.stockInfo p {
  margin: 0;
}


/* 预测天数按钮样式 */
.prediction-container {
  display: flex;
  align-items: center;
  margin-top: 10px;
}

.prediction-input {
  margin-right: 10px;
}

.prediction-input input {
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
  width: 70px;
  margin-bottom: 5px;
}

.predict-button {
  background-color: #ce3b3b;
  color: #fff;
  border: none;
  padding: 8px 16px;
  cursor: pointer;
  border-radius: 4px;
  font-size: 14px;
  margin-bottom: 5px;
}

.predict-button:hover {
  background-color: #a80707;
}

</style>