<template>
  <div class="test3" ref="root">
    <h1>{{ connected }}.{{ error }}.{{ message }}</h1>
    <input v-model="dbPath"/>
    {{ currentIndex }}<input class="log" name="student" type="range" min="1" :max="maxIndex" v-model="currentIndex" @change="click"/>
    <button @click="save">LOG IT</button>
    <div class="user" v-for="(user,k) in response" :key="k">
      <span>[{{ k }} {{ user.student }}]</span>
      <div v-for="(group, igroup) in user.groups" :key="igroup">
        <span v-for="r in group.rows" :key="r[2]" class="annotated-box">
          <img :src="'data:image/png;base64,' + r[13]" @click="show(r)"/><br/>
          <span>{{ r[14] }}</span>
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

export default {
  name: 'Annotate1',
  data () {
    return {
      dbPath: 'test-ipy2/capture.sqlite',
      currentIndex: 14,
      maxIndex: 200,
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
      let getq = (d, name, qs, min = 0, max = undefined) => qs.map(q => ({name, blob: name + '--' + q + '--' + min + '--' + max, rows: d.filter(r => r[7] === q).slice(min, max)}))
      let gets = (d, name, q, steps) => steps.map((s, i) => i).filter(i => i > 0).map(i => {
        let min = steps[i - 1]
        let max = steps[i]
        //console.log(i, min, max, d.length, d.filter(r => r[7] === q).length)
        return {
          name,
          blob: name + '--' + q + '--' + min + '--' + max,
          rows: d.filter(r => r[7] === q).slice(min, max)
        }
      })
      delete data._id
      this.response = Object.keys(data).map((u) => {
        let d = data[u]
        return {
          student: u,
          groups: [
            /*
            get(d, 'nom', 1, 0, 24),
            get(d, 'prenom', 1, 24, 47),
            get(d, 'promo', 1, 47, 50),
            get(d, 'login', 1, 50, 58),
            */
            ...gets(d, 'infos', 1, [0, 24, 47, 50, 58]),
            ...gets(d, 'q2', 2, [0, 3, 6, 9, 18, 21, 24, 27, 30, 33, 36, 39]),
            ...getq(d, 'q3a8', [3, 4, 5, 6, 7, 8]),
            ...gets(d, 'q8', 8, [0, 12, 32, 52]),
            ...gets(d, 'q9', 9, [0, 4, 12, undefined]),
            ...gets(d, 'q10', 10, [0, 18, undefined])
            /*
            get(d, 'q6', 6),
            get(d, 'q7', 7),
            get(d, 'q8a', 8, 0, 3),
            get(d, 'q8b', 8, 3, 6),
            get(d, 'q9a', 9, 0, 6),
            get(d, 'q9b', 9, 6, 12),
            get(d, 'q10', 10),
            get(d, 'q11', 11),
            get(d, 'q12', 12),
            get(d, 'q13', 13)
            */
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
      return S.bestGuess(group, this.suggestions)
    },
    isChange (u, ind) {
      console.log(ind)
      return ind === 0 || u[ind - 1][7] !== u[ind][7]
    },
    show (w) {
      console.log(w)
      this.$socket.emit('test3_show', {file: this.dbPath, rowId: w[2]})
    },
    click () {
      console.log('CLICK')
      this.$socket.emit('test3_load_all', {file: this.dbPath, _id: 'TESTID', only: this.currentIndex})
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
.annotated-box {
  display: inline-block;
}
.annotated-box span {
  border: 1px solid darkgrey;
  background: lightgrey;
  padding: 4px;
  font-weight: bold;
}
</style>
