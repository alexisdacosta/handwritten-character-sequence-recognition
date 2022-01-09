<template>
  <div>
    <h1>Digits Recognizer App</h1>
    
  </div>
</template>

<script>
import {notification} from 'vuesax'
export default {
  name: "App",
  components: {},
  mounted(){
    
    notification({
      title: 'Documentation Vuesax 4.0+',
      text: `These documents refer to the latest version of vuesax (4.0+),
      to see the documents of the previous versions you can do it here 👉 Vuesax3.x`
    });
    
  }
};
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300&display=swap');

#app {
  /* font-family: Avenir, Helvetica, Arial, sans-serif; */
  font-family: 'Poppins', sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #000000;
  margin-top: 60px;
}
</style>
