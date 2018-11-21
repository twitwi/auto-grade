<template>
  <div class="manual-label">
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    {{ currentIndex }}<input type="range" min="1" max="200" v-model="currentIndex" @change="click"/>
    <div class="user" v-for="(user,k) in response" :key="'TUTU' + k + '---' + user.length">
      <span>[{{ k }}]</span>
      <span v-for="(r,i) in user" :key="r[2]">
        <span v-if="i===0 || user[i-1][7]!==user[i][7]">{{ r[7] }}</span>
        <img :src="'data:image/png;base64,' + r[13]" @click="show(r)"/>
      </span>
      <br/>
    </div>
    <button @click="click">GO</button>
  </div>
</template>

<script>
import { mapState } from 'vuex'
// mapGetters, mapMutations, mapActions

export default {
  name: 'ManualLabel',
  data () {
    return {
      previous: [],
      next: [],
      currentIndex: 1,
      response: []
    }
  },
  created () {
    this.$options.sockets.test2rep = (data) => {
      console.log('REP')
      this.response = data
    }
  },
  computed: {
    ...mapState(['connected', 'error', 'message'])
  },
  methods: {
    clearNext () { this.next = [] },
    isChange (u, ind) {
      console.log(ind)
      return ind === 0 || u[ind - 1][7] !== u[ind][7]
    },
    show (w) {
      console.log(w)
    },
    click () {
      console.log('CLICK')
      this.$socket.emit('test2_load_all', { file: 'test3/capture.sqlite', _id: 'TESTID', only: this.currentIndex })
      console.log('CLICKED')
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
