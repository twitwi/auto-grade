<template>
  <div class="test3" ref="root">
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    {{ currentIndex }}<input class="log" name="student" type="range" min="1" max="200" v-model="currentIndex" @change="click"/>
    <button @click="save">LOG IT</button>
    <div class="user" v-for="(user,k) in response" :key="k">
      <span>[{{ k }} {{ user.student }}]</span>
      <div v-for="(group, igroup) in user.groups" :key="igroup">
        <span v-for="r in group.rows" :key="r[2]">
          <img :src="'data:image/png;base64,' + r[13]" @click="show(r)"/>
          {{ r[14] }}
        </span>
        <!--span>{{ group.name }}</span-->
        <input class="log" :name="'g:'+group.blob" type="text" :value="bestGuess(group)"/>
        <hr/>
      </div>
      <br/>
    </div>
    <button @click="click">RELOAD</button>
  </div>
</template>

<script>
import { mapState } from 'vuex'
// mapGetters, mapMutations, mapActions
import S from '../suggestions'

function argmin (arr) {
  let i = arr.length
  let min = Infinity
  let imin = -1
  while (i--) {
    if (arr[i] < min) {
      imin = i
      min = arr[i]
    }
  }
  return imin
}

export default {
  name: 'TEST3',
  data () {
    return {
      currentIndex: 1,
      response: [],
      suggestions: {
        nom: S.noms,
        prenom: S.prenoms,
        DEFAULT: S.numbers
      }
    }
  },
  created () {
    this.$options.sockets['test3rep'] = (data) => {
      console.log('REP')
      let get = (d, name, q, min = 0, max = undefined) => ({name, blob: name + '--' + q + '--' + min + '--' + max, rows: d.filter(r => r[7] === q).slice(min, max)})
      delete data._id
      this.response = Object.keys(data).map((u) => {
        let d = data[u]
        return {
          student: u,
          groups: [
            get(d, 'nom', 1, 0, 20),
            get(d, 'prenom', 1, 24, 44),
            get(d, 'q1', 7),
            get(d, 'q2', 8),
            get(d, 'q3', 9),
            get(d, 'q4', 10),
            get(d, 'q5', 11),
            get(d, 'q6', 12)
          ]
        }
      })
    }
  },
  computed: {
    ...mapState(['connected', 'error', 'message'])
  },
  methods: {
    bestGuess (group) {
      function clean (a) { return [].filter.call(a, c => c !== '_').join('') }
      var pred = clean(group.rows.map(r => r[14]).join(''))
      function dist (sug) {
        let res = 0
        let n = pred.length
        if (pred.length !== sug.length) {
          res += 3
          let nmin = Math.min(n, sug.length)
          res += 2 * (n - nmin)
          n = nmin
        }
        if (n === 0) {
          return res
        }
        while (--n >= 0) {
          res += S.cost(sug[n], pred[n])
        }
        // console.log(sug, pred, res)
        return res
      }
      let suggest = this.suggestions[group.name]
      if (suggest === undefined) suggest = this.suggestions.DEFAULT
      let dists = suggest.map(dist)
      let imax = argmin(dists)
      return suggest[imax]
    },
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
    },
    save () {
      var toLog = this.$refs.root.querySelectorAll('input.log')
      toLog = Array.prototype.slice.call(toLog)
      toLog = toLog.map(i => ({blob: i.name, value: i.value}))
      console.log(toLog)
      this.$socket.emit('test3_log', JSON.stringify(toLog) + '\n') // send a string for easier printing on the other side
      this.currentIndex = parseInt(this.currentIndex) + 1
      this.click()
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
