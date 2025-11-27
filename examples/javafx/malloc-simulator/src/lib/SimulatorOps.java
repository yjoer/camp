package lib;

public class SimulatorOps {

  public interface Ops {}

  public record AddJob(String name, int size) implements Ops {}

  public record RemoveJob(String name) implements Ops {}

  public record AddWaitingJob(String name, int size) implements Ops {}

  public record RemoveWaitingJob(String name) implements Ops {}

  public record AddPartition(Integer idx, String name, int size) implements Ops {}

  public record UpdatePartition(int idx, int size) implements Ops {}

  public record RemovePartition(int idx) implements Ops {}

  public record Allocate(int idx, String name, int size) implements Ops {}

  public record Free(int idx, String name) implements Ops {}
}
