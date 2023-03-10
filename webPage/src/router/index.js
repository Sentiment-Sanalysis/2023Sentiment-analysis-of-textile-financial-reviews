//引入路由对象
import { createRouter, createWebHistory, } from 'vue-router'


//路由数组的类型 RouteRecordRaw
// 定义一些路由
// 每个路由都需要映射到一个组件。
const routes = [{
    path: '/a',
    component: () => import('../components/a.vue')
}, {
    path: '/b',
    component: () => import('../components/b.vue')
}]



const router = createRouter({
    history: createWebHistory(),
    routes
})

//导出router
export default router