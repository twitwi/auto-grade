<template>
  <div id="app">

    <!-- picnicss navbar -->
    <nav class="demo">
      <router-link to="/">
        <img class="logo" src="./assets/logo.png" />
      </router-link>

      <!-- responsive-->
      <input id="bmenub" type="checkbox" class="show">
      <label for="bmenub" class="burger pseudo button">menu</label>

      <div class="menu">
        <router-link to="/">Home</router-link>
        <router-link to="/test2" class="pseudo button icon-picture">Raw</router-link>
        <router-link to="/annotate1" class="pseudo button icon-picture">Annotate</router-link>
        <!--a href="#" class="button icon-puzzle">Plugins</a-->
      </div>
    </nav>

    <router-view></router-view>
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>

<style>
@import "../node_modules/picnic/picnic.min.css";

#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

</style>
