<template>
  <div id="app">
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  /* font-family: 'Arial', sans-serif; */
  font-family: 'Product Sans', sans-serif;
  line-height: 1.6;
}

#app {
  min-height: 100vh;
}
</style>