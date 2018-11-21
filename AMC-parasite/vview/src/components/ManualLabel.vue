<template>
  <div class="manual-label">
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    <input v-model="projectDir"/><br/>
    {{ currentUser }}<input type="range" min="1" max="200" v-model="currentUser" @change="click"/>
    <button @click="click">GO</button>

    <div class="scroller" :style="{ 'margin-left': (400-75*currentImage)+'px'}">
      <img v-for="(i,ii) in response[currentUser]" :src="i[i.length-1]+'?'+currentUser" :key="i[i.length-1]" :class="{'current': currentImage==ii}"></img>
    </div>
    <button @keydown="alert($event)"/>
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
    }
  },
  sockets: {
    'manual-loaded-images': function(data) {
      //console.log(JSON.parse(JSON.stringify(data)));
      this.response = data;
    },
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
.scroller img { border: 1px dotted green; min-width: 75px; max-width: 75px;}
.scroller img.current { border: 1px solid black; }
</style>
