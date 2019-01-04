<template>
  <div class="exam-id">
    <h1>{{ connected }}</h1>
    <h1>{{ error }}</h1>
    <h1>{{ message }}</h1>
    <input v-model="projectDir"/><br/>
    <button @click="loadXlsx()">load parasite.xlsx</button><br/>
    <div v-for="(r,i) in rows" :key="i">
      {{ r }}
    </div>
  </div>
</template>

<script>
import config from '../customconfig'
import { mapState } from 'vuex'
// mapGetters, mapMutations, mapActions

export default {
  name: 'ExamId',
  data () {
    return {
      projectDir: config.defaultProjectDir,
      rows: []
    }
  },
  sockets: {
    'got-xlsx-structured-rows': function (data) {
      // this.logs = data.map(l => ({ d: JSON.parse(l), selected: false }))
      this.rows = data
    }
  },
  computed: {
    ...mapState(['connected', 'error', 'message'])
  },
  methods: {
    loadXlsx () {
      this.$socket.emit('xlsx-structured-rows', { pro: this.projectDir })
    },
    exportMiniset () {
      let name = new Date().toISOString().replace(/T.*/, '-' + Date.now())
      let pro = this.projectDir
      this.$socket.emit('miniset-export', { name, pro, annotations: this.logs.filter(l => l.selected).map(l => l.d) })
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.student { font-size: 150%; margin-right: 1em; }
.details { font-family: monospace; font-size: 70%; display: inline-block; overflow-x: scroll; }
</style>
