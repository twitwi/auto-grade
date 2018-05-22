<template>
  <div class="test3">
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    {{ currentIndex }}<input type="range" min="1" max="200" v-model="currentIndex" @change="click"/>
    <div class="user" v-for="(user,k) in response" :key="k">
      <span>[{{ k }} {{ user.student }}]</span>
      <div v-for="(group, igroup) in user.groups" :key="igroup">
        <span v-for="r in group" :key="r[2]">
          <img :src="'data:image/png;base64,' + r[13]" @click="show(r)"/>
          {{ r[14] }}
        </span>
      </div>
      <br/>
    </div>
    <button @click="click">GO</button>
  </div>
</template>

<script>
import { mapState } from 'vuex'
// mapGetters, mapMutations, mapActions

export default {
  name: 'TEST3',
  data () {
    return {
      currentIndex: 1,
      response: []
    }
  },
  created () {
    this.$options.sockets['test3rep'] = (data) => {
      console.log('REP')
      let get = (d, q, min = 0, max = undefined) => d.filter(r => r[7] === q).slice(min, max)
      delete data._id
      this.response = Object.keys(data).map((u) => {
        let d = data[u]
        return {
          student: u,
          groups: [
            // get(d, 1, 0, 20),
            // get(d, 1, 24, 44),
            get(d, 7),
            get(d, 8),
            get(d, 9),
            get(d, 10),
            get(d, 11),
            get(d, 12)
          ]
        }
      })
    }
  },
  computed: {
    ...mapState(['connected', 'error', 'message'])
  },
  methods: {
    isChange (u, ind) {
      console.log(ind)
      return ind === 0 || u[ind - 1][7] !== u[ind][7]
    },
    show (w) {
      console.log(w)
      this.$socket.emit('test3_show', {file: 'test3/capture.sqlite', rowId: w[2]})
    },
    click () {
      console.log('CLICK')
      this.$socket.emit('test3_load_all', {file: 'test3/capture.sqlite', _id: 'TESTID', only: this.currentIndex})
      console.log('CLICKED')
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.user {
  border: 5px solid darkgrey;
}
.user span {
  font-weight: bold;
}
a {
  color: #42b983;
}
</style>
