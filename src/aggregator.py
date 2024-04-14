import state
import bit_star_planner
import fast_marching_trees_planner
import informed_rrt_star_planner
import rrt_star_planner
import rrt_planner

import pickle
import csv


class aggregator():
    def __init__(self) -> None:
        self.pkl_file = ".\worlds\map.pkl"

    def test_bit_star(self, start, dest, n):
        results = []
        for i in range(n):
            s = state.State(start[0], start[1], None)
            d = state.State(dest[0], dest[1], None)
            pkl_file = open(".\worlds\map.pkl", 'rb')
            world = pickle.load(pkl_file)
            pkl_file.close()
            temp_bit = bit_star_planner.BITStarPlanner(world)
            plan = temp_bit.plan(s, d, 1000, 30, 50, False)
            results.append((temp_bit.cost(d, s), plan[1]))
            #print(temp_bit.cost(d, s), plan[1])

        s = ".\\results\\bit_star_" + str(start[0]) + "_"+str(start[1]) + "___" + str(dest[0]) + "_"+str(dest[1])+".csv"
        csvfile = open(s, "w+", newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)
        csvfile.close()
        
    def test_fmt(self, start, dest, n):
        results = []
        for i in range(n):
            s = state.State(start[0], start[1], None)
            d = state.State(dest[0], dest[1], None)
            pkl_file = open(".\worlds\map.pkl", 'rb')
            world = pickle.load(pkl_file)
            pkl_file.close()
            temp_fmt = fast_marching_trees_planner.FastMarchingTreesPlanner(world)
            plan = temp_fmt.plan(s, d, 30, 1000, False)
            tc = temp_fmt.total_cost(plan[0])
            results.append((tc, plan[1]))

        s = ".\\results\\fmt_" + str(start[0]) + "_"+str(start[1]) + "___" + str(dest[0]) + "_"+str(dest[1])+".csv"
        csvfile = open(s, "w+", newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)
        csvfile.close()

    def test_rrt(self, start, dest, n):
        results = []
        for i in range(n):
            s = state.State(start[0], start[1], None)
            d = state.State(dest[0], dest[1], None)
            pkl_file = open(".\worlds\map.pkl", 'rb')
            world = pickle.load(pkl_file)
            pkl_file.close()
            temp_rrt = rrt_planner.RRTPlanner(world)
            plan = temp_rrt.plan(s, d, 1000, 30, 50, False)
            results.append((temp_rrt.cost(d, s), plan[1]))
            #print(temp_bit.cost(d, s), plan[1])

        s = ".\\results\\rrt_" + str(start[0]) + "_"+str(start[1]) + "___" + str(dest[0]) + "_"+str(dest[1])+".csv"
        csvfile = open(s, "w+", newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)
        csvfile.close()
    
    def test_rrt_star(self, start, dest, n):
        results = []
        for i in range(n):
            s = state.State(start[0], start[1], None)
            d = state.State(dest[0], dest[1], None)
            pkl_file = open(".\worlds\map.pkl", 'rb')
            world = pickle.load(pkl_file)
            pkl_file.close()
            temp_rrt = rrt_star_planner.RRTStarPlanner(world)
            plan = temp_rrt.plan(s, d, 1000, 30, 50, False)
            results.append((temp_rrt.cost(d, s), plan[1]))
            #print(temp_bit.cost(d, s), plan[1])

        s = ".\\results\\rrt_star_" + str(start[0]) + "_"+str(start[1]) + "___" + str(dest[0]) + "_"+str(dest[1])+".csv"
        csvfile = open(s, "w+", newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)
        csvfile.close()

    def test_informed_rrt_star(self, start, dest, n):
        results = []
        for i in range(n):
            s = state.State(start[0], start[1], None)
            d = state.State(dest[0], dest[1], None)
            pkl_file = open(".\worlds\map.pkl", 'rb')
            world = pickle.load(pkl_file)
            pkl_file.close()
            temp_informed_rrt = informed_rrt_star_planner.InformedRRTStarPlanner(world)
            plan = temp_informed_rrt.plan(s, d, 1000, 30, 50, False)
            results.append((temp_informed_rrt.cost(d, s), plan[1]))
            #print(temp_bit.cost(d, s), plan[1])

        s = ".\\results\\informed_rrt_star_" + str(start[0]) + "_"+str(start[1]) + "___" + str(dest[0]) + "_"+str(dest[1])+".csv"
        csvfile = open(s, "w+", newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(results)
        csvfile.close()


if __name__ == "__main__":
    paths = [
        [(10, 10), (500, 500)],
        [(10, 10), (500, 150)],
        [(10, 10), (150, 250)],
        [(10, 10), (500, 250)],
        [(500, 500), (500, 200)],
        [(500, 500), (200, 500)], 
        [(10, 500), (300, 300)],
        [(10, 500), (500, 200)],
        [(600, 125), (500, 200)],
        [(600, 125), (300, 300)]

    ]
    aggr = aggregator()
    for s, d in paths:
        aggr.test_informed_rrt_star(s, d, 100)

