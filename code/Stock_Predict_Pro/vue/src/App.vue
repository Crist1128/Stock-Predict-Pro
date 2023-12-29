<template>
        <div class="logo_table">
            <div class="logo_img"></div>
            <button class="button_login">Login</button>
            <button class="button_register">Register</button>
        </div>
        <div class="sidebar">
            <div class="out">>></div>
            <div class="can_see"></div>
            <div></div>
            <div class="button_list">  
                <button @click="goHome">首页</button>
                <button @click="goGoogleFinance">谷歌财经</button>
                <button @click="goEastMoney">东方财经</button>
                <button>...</button>
                <button>...</button>
                <button>退出</button>
            </div>
        </div>
  <router-view/>
</template>

<style scoped>
@import 'assets/css/logo.css';
</style>

<script>
export default {
    methods: {
        goHome() {
            this.$router.push('/mainpage');
        },
        goGoogleFinance() {
            window.location.href = 'https://www.google.com/finance';
        },
        goEastMoney () {
            window.location.href = 'https://www.eastmoney.com/';
        }
    }
};  
</script>