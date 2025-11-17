from ga import run_evolution
import config as C

def main():
    print("Running GA with params:")
    print(f"POP_SIZE={C.POP_SIZE}, NGEN={C.NGEN}, CXPB={C.CXPB}, MUTPB={C.MUTPB}")
    pop, logbook, hof, reval = run_evolution()
    print("Best individual:", hof[0])
    print("Best fitness:", hof[0].fitness.values)
    print("Re-evaluated best test_loss:", reval)


if __name__ == "__main__":
    main()
