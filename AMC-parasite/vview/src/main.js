import Vue from 'vue'
import App from './App.vue'
import router from './router'
import VueSocketio from 'vue-socket.io'
import store from './store'

import config from './customconfig'

// Vue.config.keyCodes['letters'] = [38, 87]

Vue.config.productionTip = false
Vue.use(new VueSocketio({
  connection: config.pyConnection,
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
