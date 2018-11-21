import Vue from 'vue'
import App from './App.vue'
import router from './router'
import VueSocketio from 'vue-socket.io'
import store from './store'

// Vue.config.keyCodes['letters'] = [38, 87]

Vue.config.productionTip = false
Vue.use(new VueSocketio({
  connection: `//${window.document.domain}:5000`,
  vuex: {
    store,
    actionPrefix: 'SOCKET_',
    mutationPrefix: 'SOCKET_'
  }
}))

new Vue({
  router,
  store,
  render: h => h(App)
}).$mount('#app')
