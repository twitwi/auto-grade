<template>
  <div id="app">
    <div id="nav">
      <router-link to="/manual-label" class="pseudo button icon-picture">Annotate</router-link>
      <router-link to="/miniset-builder" class="pseudo button icon-picture">Miniset</router-link>
      <router-link to="/mass-fix" class="pseudo button icon-picture">MassFix</router-link>
      <router-link to="/exam-id" class="pseudo button icon-picture">ExamId</router-link>
      <router-link to="/ocr-questions" class="pseudo button icon-picture">OCRQuestions</router-link>
      <span>///</span>
      <router-link to="/">Home</router-link>
    </div>
    <router-view/>
  </div>
</template>

<style lang="scss">
body {
  background: black;
  filter: invert(100%);
}
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
}
#nav {
  padding: 30px;
  a {
    font-weight: bold;
    color: #2c3e50;
    &.router-link-exact-active {
      color: #42b983;
    }
  }
}
</style>
