
import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

export default new Vuex.Store({
  state: {
    connected: false,
    error: '',
    message: ''
  },
  mutations: {
    SOCKET_CONNECT (state) {
      state.connected = true
    },
    SOCKET_DISCONNECT (state) {
      state.connected = false
    },
    SOCKET_MESSAGE (state, message) {
      state.message = message
    },
    SOCKET_HELLO_WORLD (state, message) {
      state.message = message
    },
    SOCKET_ERROR (state, message) {
      state.error = message.error
    }
  }
})
