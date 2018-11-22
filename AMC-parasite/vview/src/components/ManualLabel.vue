<template>
  <div class="manual-label">
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    <input v-model="projectDir"/><br/>
    {{ currentUser }}<input type="range" min="1" max="200" v-model="currentUser" @change="click"/>
    <button @click="currentUser -- ; click()">«</button>
    <input v-model.number="currentUser"/>
    <button @click="currentUser ++ ; click()">»</button>
    <button @click="click">GO</button>

    <div class="scroller" :style="{ 'margin-left': (400-75*currentImage)+'px'}" @click.right.prevent="focus('cr/page-'+currentUser+'-'+response[currentUser][currentImage][4]+'.jpg')">
      <div v-for="(i,ii) in response[currentUser]" :key="i[i.length-1]" class="element">
        <img v-if="Math.abs(currentImage+5-ii) < 11" :src="i[i.length-1]+'?'+currentUser" :class="{'current': currentImage==ii}" />
        <span v-if="annotations[ii]" class="annotation">{{annotations[ii]}}</span>
      </div>
    </div>
    <button @keydown="keydown($event)">FOCUS</button>
    <button @click="currentImage --" title="also backspace">«</button>
    <input v-model.number="currentImage"/>
    <button @click="currentImage ++">»</button>
    <button @click="currentImage = 0">««««</button>
    <br/>
    <pre v-if="response[currentUser]">{{JSON.stringify(response[currentUser][currentImage])}}</pre>
    <br/>
    <img :src="currentFocusImage" class="focus" @click.left="currentFocusImage = ''"/>
    <!--
    <div class="user" v-for="(user,k) in response" :key="'TUTU' + k + '---' + user.length">
      <span>[{{ k }}]</span>
      <span v-for="(r,i) in user" :key="r[2]">
        <span v-if="i===0 || user[i-1][7]!==user[i][7]">{{ r[7] }}</span>
        <img :src="'data:image/png;base64,' + r[13]" @click="show(r)"/>
      </span>
      <br/>
    </div>
  -->
  </div>
</template>

<script>
import config from '../customconfig'
import { mapState } from 'vuex'
// mapGetters, mapMutations, mapActions

export default {
  name: 'ManualLabel',
  data () {
    return {
      projectDir: ',,test/2018-infospichi-3-exam-2',
      currentUser: 1,
      response: [],
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
    }
  },
  created () {
    /*
    this.$options.sockets.test2rep = (data) => {
      console.log('REP')
      this.response = data
    }
    */
  },
  computed: {
    ...mapState(['connected', 'error', 'message'])
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
        this.skipEmptys()
      } else {
        prevDef = false
        console.log(ev)
      }
      if (prevDef) {
        ev.preventDefault()
      }
    },
    focus (imPath) {
      this.currentFocusImage = config.pyConnection + '/' + this.projectDir + '/' + imPath
    },
    annotateCurrent (k) {
      this.annotations[this.currentImage] = k
      this.currentImage++
      this.skipEmptys()
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
      let toLog = [this.currentUser, this.annotations]
      this.$socket.emit('manual-log', JSON.stringify(toLog) + '\n') // send a string for easier printing on the other side
    },
    click () {
      console.log('CLICK')
      this.$socket.emit('manual-load-images', { file: this.projectDir + '/data/capture.sqlite', _id: 'TESTID', only: this.currentUser })
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
.scroller .annotation { border-bottom: 1px solid black; }
.focus { width: 100%; }
</style>
