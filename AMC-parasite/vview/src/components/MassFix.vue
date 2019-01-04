<template>
  <div class="mass-fix">
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    <input v-model="projectDir"/><br/>
    {{ currentUser }}<input type="range" min="1" max="200" v-model="currentUser" @change="click"/>
    <button @click="currentUser -- ; click()">«</button>
    <input v-model.number="currentUser"/>
    <button @click="currentUser ++ ; click()">»</button>
    <button @click="click">GO</button>

    <br/>
    <button @keydown="keydown($event)">FOCUS</button>

    <div v-for="([k, l], ind) in byLetter" :key="k" v-if="Math.abs(currentClass-ind) < 2">
      <h3>Classified as <code>{{ k }}</code></h3>
      <div class="scroller" :style="{ 'margin-left': (400-75*currentImage)+'px'}" @click.right.prevent="focus('cr/page-'+currentUser+'-'+response[currentUser][currentImage][4]+'.jpg')">
        <div v-for="(i,ii) in l" :key="i[i.length-1]" class="element">
          <img v-if="Math.abs(currentImage+5-ii) < 11" :src="svPath + i[i.length-1]+'?'+currentUser" :class="{'current': currentClass == ind && currentImage == ii}" />
          <br/>
          <span v-if="annotations[i[0]]" class="annotation">{{annotations[i[0]]}}</span>
        </div>
      </div>
    </div>

    <img :src="currentFocusImage" class="focus" @click.left="currentFocusImage = ''"/>
  </div>
</template>

<script>
import config from '../customconfig'
import { mapState } from 'vuex'
// mapGetters, mapMutations, mapActions

export default {
  name: 'MassFix',
  data () {
    return {
      projectDir: config.defaultProjectDir,
      svPath: config.pyConnection + '/',
      currentUser: 1,
      response: [],
      currentClass: 0,
      currentImage: 0,
      currentFocusImage: '',
      annotations: {}
    }
  },
  sockets: {
    'manual-loaded-images': function (data) {
      // console.log(JSON.parse(JSON.stringify(data)));
      this.response = data
      this.annotations = {}
      for (let [u, userData] of Object.entries(this.response)) {
        if (u === '_id') continue
        for (let i in userData) {
          this.$set(this.response[u][i], 0, i) // save index for annotation log
          let d = userData[i]
          if (d[d.length - 2] !== '_') {
            this.annotations[i] = d[d.length - 2]
          }
        }
      }
    }
  },
  computed: {
    ...mapState(['connected', 'error', 'message']),
    byLetter () {
      if (this.response === undefined) return {}
      if (this.response[this.currentUser] === undefined) return {}
      let res = []
      for (let c of config.classes) {
        res.push([c, this.response[this.currentUser].filter((o, i) => this.annotations[i] === c)])
      }
      return res
    }
  },
  methods: {
    keydown (ev) {
      var k = ev.key
      var prevDef = true
      if (k === 'Backspace') {
        this.currentImage--
      } else
      if (`'"!,.?-/*+=:[]()0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz`.indexOf(k) !== -1) {
        this.annotateCurrent(k)
      } else
      if (k === 'Enter') {
        this.save()
      } else
      if (k === 'Tab') {
        this.currentImage = 0
        this.currentClass++
      } else
      if (k === 'Escape') {
        this.currentImage = 0
        this.currentClass = 0
      } else
      if (k === 'ArrowRight') {
        this.currentImage++
      } else
      if (k === 'ArrowLeft') {
        this.currentImage--
      } else
      if (k === 'ArrowDown') {
        this.currentClass++
      } else
      if (k === 'ArrowUp') {
        this.currentClass--
      } else {
        prevDef = false
        console.log(ev)
      }
      if (prevDef) {
        ev.preventDefault()
      }
    },
    focus (imPath) {
      this.currentFocusImage = config.pyConnection + '/MC/' + this.projectDir + '/' + imPath
    },
    annotateCurrent (k) {
      let ind = this.byLetter[this.currentClass][1][this.currentImage][0]
      console.log(ind)
      this.$set(this.annotations, ind, k)
      this.currentImage++
    },
    skipEmptys () {
      while (this.response[this.currentUser][this.currentImage][10] === 0 && this.currentImage < this.response[this.currentUser].length) {
        this.currentImage++
      }
    },
    isChange (u, ind) {
      console.log(ind)
      return ind === 0 || u[ind - 1][7] !== u[ind][7]
    },
    show (w) {
      console.log(w)
    },
    save () {
      let toLog = { pro: this.projectDir, data: JSON.stringify([this.currentUser, this.annotations]) + '\n' }
      this.$socket.emit('manual-log', toLog) // send a string for easier printing on the other side
    },
    click () {
      console.log('CLICK')
      this.$socket.emit('manual-load-images', { pro: this.projectDir, predict: true, _id: 'MassFix', only: this.currentUser })
      console.log('CLICKED')
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.scroller { transition: margin 200ms; overflow: hidden; display: flex; }
.scroller { }
.scroller .element { min-width: 75px; }
.scroller img { box-sizing: border-box; border: 2px dotted green; min-width: 75px; max-width: 75px;}
.scroller img.current { border: 2px solid black; }
.scroller .annotation { border-bottom: 1px solid black; font-family: monospace; }
.focus { width: 100%; }
h3 { text-align: left; padding-left: 150px; }
</style>
